import numpy as np
import pandas as pd


def create_test_data(start_date, symbols, n):
    np.random.seed(42)
    num_symbols = len(symbols)
    return_values_cycle = np.array([0.1, 0.2])  # 预设的收益率循环值

    trade_time_range = pd.date_range(start=start_date, periods=n, freq='B')
    index = pd.MultiIndex.from_product([trade_time_range, symbols],
                                       names=['trade_time', 'code'])

    total_rows = n * num_symbols  # DataFrame的总行数

    base_price_paths = np.cumsum(np.random.randn(n, num_symbols), axis=0) + 100

    base_price_flat = base_price_paths.flatten()

    open_price = base_price_flat + np.random.randn(total_rows) * 0.5
    close_price = base_price_flat + np.random.randn(total_rows) * 0.5

    max_oc = np.maximum(open_price, close_price)
    min_oc = np.minimum(open_price, close_price)

    high_add = np.random.rand(
        total_rows) * 0.3 + 0.01  # 加一个小的正数确保high > max_oc
    low_sub = np.random.rand(total_rows) * 0.3 + 0.01  # 减一个小的正数确保low < min_oc

    high_price = max_oc + high_add
    low_price = min_oc - low_sub

    current_high_gt_low = high_price > low_price
    low_price = np.where(current_high_gt_low, low_price, high_price - 0.01)

    volume = np.random.randint(1000, 10000, size=total_rows)
    amount = volume * close_price

    num_return_values = len(return_values_cycle)
    single_stock_returns = np.tile(return_values_cycle,
                                   n // num_return_values + 1)[:n]

    returns = np.repeat(single_stock_returns, num_symbols)

    data_dict = {
        'open': open_price.round(2),
        'high': high_price.round(2),
        'low': low_price.round(2),
        'close': close_price.round(2),
        'volume': volume,
        'amount': amount.round(2),
        'return': returns  # 收益率已经是正确的形状和值
    }
    total_data = pd.DataFrame(data_dict, index=index)
    # 调整列顺序以匹配您的期望
    total_data = total_data[[
        'open', 'high', 'low', 'close', 'volume', 'amount', 'return'
    ]]

    return total_data
