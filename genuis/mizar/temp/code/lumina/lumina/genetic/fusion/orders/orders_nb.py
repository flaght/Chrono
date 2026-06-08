import pdb
import pandas as pd
import numpy as np
from numba import njit
from collections import namedtuple
# from typing import List # Optional: for type hinting

# Assuming OrderTuple and RESULT_CODE_MAP are defined as provided
OrderTuple = namedtuple('OrderTuple', [
    'buy_time', 'buy_price', 'buy_cnt', 'sell_time', 'sell_price', 'sell_type',
    'expect_direction', 'buy_symbol', 'commission', 'slippage'
])


@njit
def determine_trade_result_numba(entry_price: float, exit_price: float,
                                 direction: int) -> int:
    # Returns code: 0: unknown_price, 1: win, 2: loss, 3: even, 4: unknown_direction
    if np.isnan(entry_price) or np.isnan(exit_price):
        return 0
    if direction == 1:  # Long
        if exit_price > entry_price: return 1  # win
        elif exit_price < entry_price: return 2  # loss
        else: return 3  # even
    elif direction == -1:  # Short
        if exit_price < entry_price: return 1  # win
        elif exit_price > entry_price: return 2  # loss
        else: return 3  # even
    return 4  # unknown_direction


RESULT_CODE_MAP = {
    0: 'keep',
    1: 'win',
    2: 'loss',
    3: 'even',
    4: 'unknown_direction'
}


@njit
def _position_next_order_numba_core(
    pos_values: np.ndarray, 
    trade_action_times_ns: np.ndarray, 
    trade_action_prices: np.ndarray
) -> list: # Numba will compile this to return a list of tuples
    orders_data = [] 

    has_active_trade = False
    active_entry_time_ns = np.int64(0) 
    active_entry_price = np.float64(0.0) 
    active_direction = np.int32(0) 

    n = len(pos_values)
    if n == 0:
        return orders_data

    for i in range(n):
        current_pos_target = pos_values[i]
        potential_trade_exec_time_ns = trade_action_times_ns[i]
        potential_trade_exec_price = trade_action_prices[i]

        # pd.NaT.value is np.iinfo(np.int64).min which is negative.
        # A valid timestamp converted to ns will be positive (epoch nanoseconds).
        is_execution_possible = (potential_trade_exec_time_ns > np.int64(0)) and \
                                (not np.isnan(potential_trade_exec_price))

        previous_pos_target = np.float64(0.0)
        if i > 0:
            previous_pos_target = pos_values[i - 1]

        if not has_active_trade:
            if previous_pos_target == 0.0 and current_pos_target != 0.0:
                if is_execution_possible:
                    active_entry_time_ns = potential_trade_exec_time_ns
                    active_entry_price = potential_trade_exec_price
                    new_direction = np.int32(0)
                    if current_pos_target > 0.0: new_direction = 1
                    elif current_pos_target < 0.0: new_direction = -1

                    if new_direction != 0: 
                        active_direction = new_direction
                        has_active_trade = True
        elif has_active_trade:
            is_closing_signal = (current_pos_target == 0.0
                                 and previous_pos_target == active_direction)

            is_reversing_signal = (current_pos_target == -active_direction
                                   and previous_pos_target == active_direction)

            if is_closing_signal:
                exit_time_ns = active_entry_time_ns 
                exit_price = np.nan 
                if is_execution_possible:
                    exit_time_ns = potential_trade_exec_time_ns
                    exit_price = potential_trade_exec_price

                result_code = determine_trade_result_numba(
                    active_entry_price, exit_price, active_direction)
                orders_data.append((
                    active_entry_time_ns,
                    active_entry_price,
                    np.int64(1),  # buy_cnt is assumed 1, ensure type for Numba tuple consistency
                    exit_time_ns,
                    exit_price,
                    result_code, # int
                    active_direction)) # int32
                has_active_trade = False
                active_direction = np.int32(0) 

            elif is_reversing_signal:
                old_leg_exit_time_ns = active_entry_time_ns
                old_leg_exit_price = np.nan
                if is_execution_possible:
                    old_leg_exit_time_ns = potential_trade_exec_time_ns
                    old_leg_exit_price = potential_trade_exec_price

                result_code_old_leg = determine_trade_result_numba(
                    active_entry_price, old_leg_exit_price, active_direction)
                orders_data.append(
                    (active_entry_time_ns, active_entry_price, np.int64(1),
                     old_leg_exit_time_ns, old_leg_exit_price,
                     result_code_old_leg, active_direction))

                if is_execution_possible:
                    active_entry_time_ns = potential_trade_exec_time_ns
                    active_entry_price = potential_trade_exec_price
                    active_direction = np.int32(-active_direction)
                else:
                    has_active_trade = False
                    active_direction = np.int32(0)
    return orders_data


def position_next_order(pos_data: pd.DataFrame,
                        market_data: pd.DataFrame,
                        commission: float = 0.0,
                        slippage: float = 0.0) -> list: # Python type hint: List[OrderTuple]
    orders = []
    if pos_data.empty:
        return orders

    # --- Symbol extraction (using your original logic) ---
    # This part remains complex; if it becomes a bottleneck with many different small DataFrames,
    # it could be further reviewed. For large DataFrames, its one-off cost is less critical.
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
            idx_of_pos = pos_level_0.get_loc('pos')
            if isinstance(idx_of_pos, slice):
                symbol = pos_data.columns.get_level_values(1)[idx_of_pos.start]
            elif isinstance(idx_of_pos, (int, np.integer)): # np.integer covers numpy int types
                symbol = pos_data.columns.get_level_values(1)[idx_of_pos]
            else:  # bool array
                true_indices = np.where(idx_of_pos)[0]
                if len(true_indices) > 0:
                    symbol = pos_data.columns.get_level_values(1)[
                        true_indices[0]]
        except (IndexError, KeyError):
            pass
    
    if symbol is None:
        # Attempt a simpler inference if pos_data column is 'pos' and market_data has one column
        if 'pos' in pos_data.columns and len(pos_data.columns) == 1 and \
           len(market_data.columns) == 1 and not isinstance(market_data.columns, pd.MultiIndex):
            symbol = market_data.columns[0] # Assume market data's single column name is the symbol
        else:
            raise ValueError(
                f"Numba: Could not automatically determine symbol. pos_data cols: {pos_data.columns}, market_data cols: {market_data.columns}"
            )

    # Determine pos_col_name
    if isinstance(pos_data.columns, pd.MultiIndex):
        pos_col_name = ('pos', symbol)
    else: # Simple Index
        pos_col_name = 'pos' # Assuming 'pos' if simple index and symbol was derived e.g. from market_data

    # Determine price_col_name
    if isinstance(market_data.columns, pd.MultiIndex):
        price_col_name = ('close', symbol)
    else:  # market_data.columns is a simple Index
        price_col_name = symbol # Assumes market data column is named after the symbol

    # Validate column names
    if pos_col_name not in pos_data.columns:
        raise ValueError(
            f"Numba: Position column {pos_col_name} not found in pos_data ({pos_data.columns})."
        )
    if price_col_name not in market_data.columns:
         # Try 'close' as a fallback if symbol name isn't in simple market_data columns
        if not isinstance(market_data.columns, pd.MultiIndex) and 'close' in market_data.columns:
            price_col_name = 'close'
        else:
            raise ValueError(
                f"Numba: Price column {price_col_name} (or 'close') not found in market_data ({market_data.columns})."
            )

    # Ensure DatetimeIndex
    if not isinstance(pos_data.index, pd.DatetimeIndex):
        pos_data.index = pd.to_datetime(pos_data.index)
    if not isinstance(market_data.index, pd.DatetimeIndex):
        market_data.index = pd.to_datetime(market_data.index)

    # Data alignment and preparation
    pos_data_sorted = pos_data.sort_index()
    market_data_sorted = market_data.sort_index()

    # Extract pos_values
    pos_values_np = pos_data_sorted[pos_col_name].fillna(0).values
    
    # --- Prepare trade_action_times and trade_action_prices (OPTIMIZED) ---
    num_signals = len(pos_data_sorted)
    trade_action_times_ns_np = np.full(num_signals, np.iinfo(np.int64).min, dtype=np.int64) # pd.NaT.value
    trade_action_prices_np = np.full(num_signals, np.nan, dtype=np.float64)

    if num_signals > 0 and not market_data_sorted.empty:
        signal_times_np = pos_data_sorted.index.values # np.datetime64 array
        market_times_np = market_data_sorted.index.values # np.datetime64 array
        market_prices_np = market_data_sorted[price_col_name].values

        # Find indices in market_times_np for each signal_time
        # 'right' side means insertion point to maintain order, elements to right are > value
        # This gives the index of the first market_time > signal_time
        exec_indices = market_times_np.searchsorted(signal_times_np, side='right')

        # Create a boolean mask for valid execution indices found by searchsorted
        # These are indices into market_data arrays that are within bounds.
        valid_market_indices_mask = exec_indices < len(market_times_np)
        
        # Get the indices of signals that have a valid corresponding market execution
        signal_indices_with_valid_exec = np.where(valid_market_indices_mask)[0]
        
        # Get the market data indices to use for these valid executions
        market_indices_for_valid_exec = exec_indices[valid_market_indices_mask]

        if len(signal_indices_with_valid_exec) > 0:
            # Populate with actual execution times (as int64 ns) and prices
            trade_action_times_ns_np[signal_indices_with_valid_exec] = \
                market_times_np[market_indices_for_valid_exec].astype(np.int64)
            trade_action_prices_np[signal_indices_with_valid_exec] = \
                market_prices_np[market_indices_for_valid_exec]

    # Call the Numba core function
    raw_orders_data = _position_next_order_numba_core(
        pos_values_np, trade_action_times_ns_np, trade_action_prices_np)

    # Convert raw data back to OrderTuple objects
    original_tz = pos_data.index.tz  # Preserve original timezone if any
    
    # Robustly get latest_price
    latest_price = np.nan
    if not market_data_sorted.empty and price_col_name in market_data_sorted:
        price_series = market_data_sorted[price_col_name]
        if not price_series.empty:
            latest_price = price_series.iloc[-1]
            
    nan_int64_min = np.iinfo(np.int64).min # Cache for slightly faster access in loop

    for data_tuple in raw_orders_data:
        buy_time_ns, buy_price, buy_cnt, sell_time_ns, sell_price_raw, result_code, expect_dir = data_tuple

        buy_time_ts = pd.NaT if buy_time_ns == nan_int64_min else pd.Timestamp(
                buy_time_ns, unit='ns', tz=original_tz)
        sell_time_ts = pd.NaT if sell_time_ns == nan_int64_min else pd.Timestamp(
                sell_time_ns, unit='ns', tz=original_tz)
        
        # If sell_type is 'keep' (result_code 0), use latest_price for sell_price
        current_sell_price = sell_price_raw
        if RESULT_CODE_MAP.get(result_code) == 'keep': # Use .get for safety
            current_sell_price = latest_price

        orders.append(
            OrderTuple(
                buy_time=buy_time_ts,
                buy_price=buy_price,
                buy_cnt=int(buy_cnt), # Ensure buy_cnt is standard Python int
                sell_time=sell_time_ts,
                sell_price=current_sell_price,
                sell_type=RESULT_CODE_MAP.get(result_code, 'unknown_code'), # Safety for unknown codes
                expect_direction=int(expect_dir), # Ensure expect_dir is standard Python int
                buy_symbol=symbol,
                commission=commission,
                slippage=slippage))
    return orders