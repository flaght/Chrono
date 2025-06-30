import pdb
import pandas as pd
from lumina.empyrical.orders.adapter import OrderTuple # Ensure this import works

def determine_trade_result(entry_price, exit_price, direction):
    """
    根据入场价、出场价和方向判断交易是盈利还是亏损。
    """
    if pd.isna(entry_price) or pd.isna(exit_price): # 如果价格无效，无法判断
        return 'unknown_price'
    if direction == 1: # 多头
        if exit_price > entry_price:
            return 'win'
        elif exit_price < entry_price:
            return 'loss'
        else:
            return 'even'
    elif direction == -1: # 空头
        if exit_price < entry_price:
            return 'win'
        elif exit_price > entry_price:
            return 'loss'
        else:
            return 'even'
    return 'unknown_direction'

def position_next_order(pos_data, market_data, commission, slippage):
    """
    将持仓数据转换为订单列表。
    sell_type 表示：win 或loss，即盈利单 还是亏损单。
    expect_direction 使用1 表示多头单 -1表示空头单。
    交易发生在上一个bar的收盘，使用上一个bar的收盘价。
    若数据结束仍有持仓，则以最后一个市场数据价格平仓。
    """
    orders = []
    active_trade = None

    if pos_data.empty:
        return orders # 如果持仓数据为空，直接返回

    symbol = None
    # --- (Symbol extraction logic - 与之前相同，保持不变) ---
    if not pos_data.columns.empty and isinstance(
            pos_data.columns, pd.MultiIndex) and len(
                pos_data.columns.levels) > 1:
        try:
            symbol = pos_data.columns.get_level_values(1)[0]
        except IndexError:
            if not market_data.columns.empty and isinstance(
                    market_data.columns, pd.MultiIndex) and len(
                        market_data.columns.levels) > 1:
                try:
                    symbol = market_data.columns.get_level_values(1)[0]
                except IndexError:
                    pass
    
    if symbol is None and isinstance(pos_data.columns, pd.MultiIndex) and ('pos' in pos_data.columns.levels[0]):
        try:
            pos_level_0 = pos_data.columns.get_level_values(0)
            idx_of_pos = pos_level_0.get_loc('pos')
            if isinstance(idx_of_pos, slice):
                 symbol = pos_data.columns.get_level_values(1)[idx_of_pos.start]
            elif isinstance(idx_of_pos, int):
                 symbol = pos_data.columns.get_level_values(1)[idx_of_pos]
            else:
                symbol = pos_data.columns.get_level_values(1)[idx_of_pos][0]
        except (IndexError, KeyError):
             raise ValueError("Could not automatically determine symbol. Columns: {}".format(pos_data.columns))

    if symbol is None:
        raise ValueError("Could not automatically determine symbol from input DataFrames.")

    pos_col_name = ('pos', symbol)
    price_col_name = ('close', symbol)

    if pos_col_name not in pos_data.columns:
        raise ValueError(f"持仓列 {pos_col_name} 在 pos_data ({pos_data.columns}) 中未找到。")
    if price_col_name not in market_data.columns:
        raise ValueError(f"价格列 {price_col_name} 在 market_data ({market_data.columns}) 中未找到。")

    if not isinstance(pos_data.index, pd.DatetimeIndex) or not isinstance(market_data.index, pd.DatetimeIndex):
        print("Warning: pos_data or market_data index is not a DatetimeIndex.")

    try:
        # 对齐市场数据，确保每个 pos_data 的时间点都有对应的市场数据（可能通过 ffill 填充）
        aligned_market_data = market_data.reindex(pos_data.index, method='ffill')
        # 额外确保 market_data 至少覆盖到 pos_data 的最后一个点，如果 market_data 更短，
        # reindex 时会用 NaN 填充，ffill 也无法填充末尾的 NaN。
        # 但如果 market_data 本身就比 pos_data 短，这里我们依赖 ffill 的结果。
        # 最后的平仓逻辑会处理最后一个点的价格。
    except Exception as e:
        print(f"Error during market_data.reindex: {e}. Using original market_data.")
        aligned_market_data = market_data # Fallback, might lead to KeyErrors later

    # --- (Loop for processing trades - 与之前版本类似，除了active_trade的更新) ---
    for i in range(len(pos_data)):
        current_time = pos_data.index[i]
        current_pos = pos_data.loc[current_time, pos_col_name]

        previous_time = None
        previous_pos = 0.0
        previous_price = float('nan')

        if i > 0:
            previous_time = pos_data.index[i-1]
            previous_pos = pos_data.loc[previous_time, pos_col_name]
            try:
                previous_price = aligned_market_data.loc[previous_time, price_col_name]
                if pd.isna(previous_price):
                    print(f"警告: {previous_time} 的 {price_col_name} 价格为 NaN (用于交易)。")
            except KeyError:
                print(f"警告: {previous_time} 未找到 {price_col_name} 的价格 (用于交易)。")
                previous_price = float('nan')
        
        current_market_price = float('nan')
        try:
            current_market_price = aligned_market_data.loc[current_time, price_col_name]
            if pd.isna(current_market_price):
                 print(f"警告: {current_time} 的 {price_col_name} 市场价格为 NaN。")
        except KeyError:
            print(f"警告: {current_time} 未找到 {price_col_name} 的市场价格。")

        if active_trade is None:
            if previous_pos == 0.0 and current_pos != 0.0 and previous_time is not None and not pd.isna(previous_price):
                if current_pos == 1.0:
                    active_trade = {'entry_time': previous_time, 'entry_price': previous_price, 'direction': 1}
                elif current_pos == -1.0:
                    active_trade = {'entry_time': previous_time, 'entry_price': previous_price, 'direction': -1}
            elif previous_pos == 0.0 and current_pos != 0.0:
                 print(f"信息: 在 {current_time} 检测到从0到 {current_pos} 的持仓变化，但无法使用前一时刻 {previous_time} 的价格 {previous_price} 开仓。")
        
        elif active_trade is not None:
            expected_direction = active_trade['direction']
            entry_price = active_trade['entry_price']
            entry_time = active_trade['entry_time']

            is_closing_position = (expected_direction == 1 and previous_pos == 1.0 and current_pos == 0.0) or \
                                  (expected_direction == -1 and previous_pos == -1.0 and current_pos == 0.0)
            is_reversing_position = (expected_direction == 1 and previous_pos == 1.0 and current_pos == -1.0) or \
                                    (expected_direction == -1 and previous_pos == -1.0 and current_pos == 1.0)

            if is_closing_position and previous_time is not None and not pd.isna(previous_price):
                trade_result = determine_trade_result(entry_price, previous_price, expected_direction)
                order_data = {
                    'buy_time': entry_time, 'buy_price': entry_price, 'buy_cnt': 1,
                    'sell_time': previous_time, 'sell_price': previous_price,
                    'sell_type': trade_result, 'expect_direction': expected_direction,
                    'buy_symbol': symbol
                }
                orders.append(OrderTuple(**order_data))
                active_trade = None
            
            elif is_reversing_position:
                # 1. Close old position
                if previous_time is not None and not pd.isna(previous_price):
                    trade_result = determine_trade_result(entry_price, previous_price, expected_direction)
                    order_data_close = {
                        'buy_time': entry_time, 'buy_price': entry_price, 'buy_cnt': 1,
                        'sell_time': previous_time, 'sell_price': previous_price,
                        'sell_type': trade_result, 'expect_direction': expected_direction,
                        'buy_symbol': symbol
                    }
                    orders.append(OrderTuple(**order_data_close))
                else:
                    print(f"警告: 在 {current_time} 进行反手操作时，无法使用前一时刻 {previous_time} 的价格 {previous_price} 平掉旧仓位。")

                # 2. Open new position
                if not pd.isna(current_market_price):
                    new_direction = -1 if expected_direction == 1 else 1
                    active_trade = {
                        'entry_time': current_time,
                        'entry_price': current_market_price,
                        'direction': new_direction
                    }
                else:
                    print(f"警告: 在 {current_time} 进行反手操作时，无法获取有效的当前市场价格 {current_market_price} 来建立新仓位。")
                    active_trade = None
            # (Optional: Add handling for other unhandled position changes if necessary)

    # --- End of loop: Check for and close any open positions ---
    if active_trade is not None:
        print(f"信息: 数据处理结束，仍有持仓: {active_trade}. 尝试以最后一个市场价格平仓。")
        
        # 获取最后一个 pos_data 的时间点作为平仓时间参考
        # 理论上，平仓发生在最后一个已知仓位信号 (pos_data.index[-1]) 之后，
        # 使用该信号K线 (pos_data.index[-1]) 的收盘价。
        final_close_time = pos_data.index[-1]
        final_close_price = float('nan')

        try:
            # 尝试获取最后一个时间点的市场价格
            # 注意：aligned_market_data 是基于 pos_data.index reindex 的，所以可以直接用 final_close_time
            final_close_price = aligned_market_data.loc[final_close_time, price_col_name]
            if pd.isna(final_close_price):
                print(f"警告: 最后一个时间点 {final_close_time} 的市场价格为 NaN。")
                # 可以尝试向前查找最后一个有效价格，但这里为了简化，如果最后一个点是NaN，平仓会用NaN价
                # 一个更稳健的做法可能是从 aligned_market_data[price_col_name].last_valid_index() 获取价格
                # 但这可能与 final_close_time 不符。
                # 策略决定：如果最后一个点的价格是NaN，如何处理？
                # 1. 用NaN（如当前代码）
                # 2. 寻找aligned_market_data中最后一个有效价格（可能时间戳不同）
                # 3. 标记为无法平仓
                #
                # 这里假设我们必须在 final_close_time 这个“时刻”平仓，如果价格是NaN，那就是NaN
                # 如果希望用最后一个有效价格，代码会复杂些：
                # last_valid_price_idx = aligned_market_data[price_col_name].last_valid_index()
                # if last_valid_price_idx is not None:
                #     final_close_price = aligned_market_data.loc[last_valid_price_idx, price_col_name]
                #     final_close_time = last_valid_price_idx # 时间也应更新
                # else:
                #     print("警告: market_data中没有有效的价格用于最终平仓。")

        except KeyError:
            print(f"警告: 在最后一个时间点 {final_close_time} 未找到市场价格。")
            # 这种情况不应该发生，因为 aligned_market_data 是 reindex 过的

        # 如果 final_close_price 仍然是 NaN，平仓单的 sell_price 会是 NaN
        # 这会导致 determine_trade_result 返回 'unknown_price'
        
        expected_direction = active_trade['direction']
        entry_price = active_trade['entry_price']
        entry_time = active_trade['entry_time']

        trade_result = determine_trade_result(entry_price, final_close_price, expected_direction)
        
        order_data_final_close = {
            'buy_time': entry_time,
            'buy_price': entry_price,
            'buy_cnt': 1,
            'sell_time': final_close_time, # 使用最后一个 pos_data 时间戳作为平仓时间
            'sell_price': final_close_price,
            'sell_type': trade_result,
            'expect_direction': expected_direction,
            'buy_symbol': symbol
        }
        orders.append(OrderTuple(**order_data_final_close))
        active_trade = None # 标记已平仓

    return orders