import pdb
import pandas as pd
import numpy as np
# from numba import njit # No longer needed for these functions
from collections import namedtuple

# Import the compiled Cython module
import order_cy

OrderTuple = namedtuple('OrderTuple', [
    'buy_time', 'buy_price', 'buy_cnt', 'sell_time', 'sell_price', 'sell_type',
    'expect_direction', 'buy_symbol', 'commission', 'slippage'
])

# determine_trade_result_numba is now internal to Cython as _determine_trade_result_cy
# _position_next_order_numba_core is now internal to Cython as _core_logic_cy

RESULT_CODE_MAP = {
    0: 'keep', # Was 'unknown_price' in Numba version, 'keep' based on usage
    1: 'win',
    2: 'loss',
    3: 'even',
    4: 'unknown_direction'
}
# Ensure RESULT_CODE_MAP aligns with return codes from _determine_trade_result_cy


def position_next_order(pos_data: pd.DataFrame,
                        market_data: pd.DataFrame,
                        commission: float = 0.0,
                        slippage: float = 0.0,
                        name='close') -> list: # Python type hint: List[OrderTuple]
    orders = []
    if pos_data.empty:
        return orders

    # --- Symbol extraction (user's original robust logic) ---
    symbol = None
    if not pos_data.columns.empty and isinstance(
            pos_data.columns, pd.MultiIndex) and len(
                pos_data.columns.levels) > 1:
        try:
            symbol = pos_data.columns.get_level_values(1)[0]
        except IndexError:
            pass
    if symbol is None and not market_data.columns.empty and isinstance(
            market_data.columns, pd.MultiIndex) and len(
                market_data.columns.levels) > 1:
        try:
            symbol = market_data.columns.get_level_values(1)[0]
        except IndexError:
            pass
    if symbol is None and isinstance(pos_data.columns, pd.MultiIndex) and \
       len(pos_data.columns.levels) > 1 and 'pos' in pos_data.columns.get_level_values(0).unique():
        try:
            pos_level_0 = pos_data.columns.get_level_values(0)
            idx_of_pos = pos_level_0.get_loc('pos') # Can be slice, int, or bool array
            if isinstance(idx_of_pos, slice): # pragma: no cover
                symbol_idx = idx_of_pos.start # Take the first one if it's a slice
                symbol = pos_data.columns.get_level_values(1)[symbol_idx]
            elif isinstance(idx_of_pos, (int, np.integer)):
                symbol = pos_data.columns.get_level_values(1)[idx_of_pos]
            elif isinstance(idx_of_pos, np.ndarray) and idx_of_pos.dtype == bool: # Bool array
                true_indices = np.where(idx_of_pos)[0]
                if len(true_indices) > 0:
                    symbol = pos_data.columns.get_level_values(1)[true_indices[0]]
            else: # pragma: no cover
                 pass # Could be an array of integers if 'pos' appears multiple times
        except (IndexError, KeyError):
            pass
    
    if symbol is None:
        if 'pos' in pos_data.columns and len(pos_data.columns) == 1 and \
           len(market_data.columns) == 1 and not isinstance(market_data.columns, pd.MultiIndex):
            symbol = market_data.columns[0]
        else:
            raise ValueError(
                f"Cython: Could not automatically determine symbol. pos_data cols: {pos_data.columns}, market_data cols: {market_data.columns}"
            )

    # Determine pos_col_name
    if isinstance(pos_data.columns, pd.MultiIndex):
        pos_col_name = ('pos', symbol)
    else: 
        pos_col_name = 'pos' 

    # Determine price_col_name
    if isinstance(market_data.columns, pd.MultiIndex):
        price_col_name = (name, symbol)
    else:
        price_col_name = symbol 

    # Validate column names
    if pos_col_name not in pos_data.columns:
        raise ValueError(
            f"Cython: Position column {pos_col_name} not found in pos_data ({pos_data.columns})."
        )
    if price_col_name not in market_data.columns:
        if not isinstance(market_data.columns, pd.MultiIndex) and name in market_data.columns:
            price_col_name = name # Fallback for simple index market data
        else:
            raise ValueError(
                f"Cython: Price column {price_col_name} (or {name}) not found in market_data ({market_data.columns})."
            )

    if not isinstance(pos_data.index, pd.DatetimeIndex):
        pos_data.index = pd.to_datetime(pos_data.index)
    if not isinstance(market_data.index, pd.DatetimeIndex):
        market_data.index = pd.to_datetime(market_data.index)

    pos_data_sorted = pos_data.sort_index()
    market_data_sorted = market_data.sort_index()

    pos_values_np = pos_data_sorted[pos_col_name].fillna(0).values.astype(np.float64)
    
    num_signals = len(pos_data_sorted)
    trade_action_times_ns_np = np.full(num_signals, np.iinfo(np.int64).min, dtype=np.int64)
    trade_action_prices_np = np.full(num_signals, np.nan, dtype=np.float64)

    if num_signals > 0 and not market_data_sorted.empty:
        signal_times_dt64_np = pos_data_sorted.index.values 
        market_times_dt64_np = market_data_sorted.index.values
        market_prices_val_np = market_data_sorted[price_col_name].values.astype(np.float64) # Ensure float

        exec_indices = market_times_dt64_np.searchsorted(signal_times_dt64_np, side='right')
        valid_market_indices_mask = exec_indices < len(market_times_dt64_np)
        signal_indices_with_valid_exec = np.where(valid_market_indices_mask)[0]
        market_indices_for_valid_exec = exec_indices[valid_market_indices_mask]

        if len(signal_indices_with_valid_exec) > 0:
            trade_action_times_ns_np[signal_indices_with_valid_exec] = \
                market_times_dt64_np[market_indices_for_valid_exec].astype(np.int64)
            trade_action_prices_np[signal_indices_with_valid_exec] = \
                market_prices_val_np[market_indices_for_valid_exec]

    # Ensure arrays are C-contiguous for optimal Cython memoryview performance
    pos_values_np_c = np.ascontiguousarray(pos_values_np)
    trade_action_times_ns_np_c = np.ascontiguousarray(trade_action_times_ns_np)
    trade_action_prices_np_c = np.ascontiguousarray(trade_action_prices_np)

    # Call the Cython core function
    raw_orders_data = order_cy._core_logic_cy(
        pos_values_np_c, trade_action_times_ns_np_c, trade_action_prices_np_c)

    original_tz = pos_data_sorted.index.tz
    
    latest_price = np.nan
    if not market_data_sorted.empty and price_col_name in market_data_sorted:
        price_series = market_data_sorted[price_col_name]
        if not price_series.empty:
            # Ensure latest_price is a float, not a Series/DataFrame element
            val = price_series.iloc[-1]
            if isinstance(val, (pd.Series, pd.DataFrame)): # pragma: no cover
                latest_price = float(val.iloc[0]) if not val.empty else np.nan
            else:
                latest_price = float(val)
            
    nat_int64_val = np.iinfo(np.int64).min

    for data_tuple in raw_orders_data: # This is a list of Python tuples from Cython
        buy_time_ns, buy_price, buy_cnt_raw, sell_time_ns, sell_price_raw, result_code, expect_dir = data_tuple

        buy_time_ts = pd.NaT if buy_time_ns == nat_int64_val else pd.Timestamp(
                buy_time_ns, unit='ns', tz=original_tz)
        sell_time_ts = pd.NaT if sell_time_ns == nat_int64_val else pd.Timestamp(
                sell_time_ns, unit='ns', tz=original_tz)
        
        current_sell_price = sell_price_raw
        # Code 0 from _determine_trade_result_cy means "unknown_price" (effectively 'keep' if still active)
        if RESULT_CODE_MAP.get(result_code) == 'keep': 
            if not np.isnan(latest_price): # Only use if latest_price is valid
                 current_sell_price = latest_price
            # else keep sell_price_raw (which would be NaN if execution wasn't possible, or entry price for 0-duration trades)

        orders.append(
            OrderTuple(
                buy_time=buy_time_ts,
                buy_price=buy_price,
                buy_cnt=int(buy_cnt_raw), 
                sell_time=sell_time_ts,
                sell_price=current_sell_price,
                sell_type=RESULT_CODE_MAP.get(result_code, f'unknown_code_{result_code}'),
                expect_direction=int(expect_dir),
                buy_symbol=symbol,
                commission=commission,
                slippage=slippage))
    return orders