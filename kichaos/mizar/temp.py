import pandas as pd
import numpy as np


def create_data():

    # --- 1. 定义参数 ---
    num_rows = 5000
    columns = [
        'trade_time', 'code', 'nxt1_ret', 'tn003_5_5_10_15_1',
        'tn003_10_5_10_15_1', 'tn003_5_5_10_15_0', 'tn003_10_5_10_15_0',
        'tn004_5_10_1', 'tn004_10_15_1', 'tn004_5_10_0', 'oi003_5_10_1',
        'oi003_5_10_0', 'oi004_5_10_0', 'oi022_5_10_1', 'oi023_5_10_1',
        'oi023_10_15_1', 'oi023_5_10_0', 'oi031_5_10_1', 'oi032_5_5_10_1',
        'oi032_10_10_15_1', 'oi033_5_10_1', 'oi034_5_10_1', 'oi034_5_10_0',
        'oi037_5_10_1', 'oi037_10_15_1', 'oi037_5_10_0', 'oi039_5_10_1',
        'oi039_5_10_0', 'oi042_5_10_1', 'oi042_10_15_1', 'oi042_5_10_0',
        'tc003_5_10_0', 'tc003_5_10_1', 'tc005_5_5_10_0', 'tc006_5_10_0',
        'tc006_5_10_1', 'tc007_5_10_1', 'tc008_5_10_0', 'tc008_5_10_1',
        'tc012_5_5_10_0', 'tc013_5_10_0', 'tc013_5_10_1', 'tc014_5_5_10_0',
        'tc015_10_15_1', 'tc015_5_10_0', 'tc015_5_10_1', 'dv002_5_10_1',
        'dv002_5_10_0', 'db001_5_10_1', 'db005_5_10_1', 'db007_5_10_0',
        'db007_5_10_1', 'cj010_5_10_1', 'cj011_10_15_1', 'cj011_5_10_0',
        'cj011_5_10_1', 'tv005_5_10_0', 'tv005_5_10_1', 'tv017_5_10_0',
        'tv017_5_10_1', 'tv018_5_10_0', 'tv018_5_10_1', 'ixy006_5_10_1',
        'ixy007_5_10_1', 'ixy007_10_15_1', 'ixy007_5_10_0', 'ixy008_5_10_1',
        'ixy014_5_10_1', 'ixy015_5_10_1', 'fz002_5_10_1', 'gd002_10_15_1',
        'gd002_5_10_0', 'gd002_5_10_1', 'rv001_5_10_1_2', 'rv001_5_10_0_2',
        'rv004_10_15_0_2', 'rv005_5_10_1_1', 'rv005_5_10_1_2',
        'rv005_10_15_1_2', 'rv005_5_10_0_2', 'rv005_10_15_0_2',
        'rv006_5_10_1_2', 'rv006_10_15_1_2', 'rv006_5_10_0_2',
        'rv011_75_5_10_1', 'rv011_75_5_10_0', 'rv011_25_5_10_1',
        'rv011_25_5_10_0', 'rv012_25_5_10_1', 'rv012_25_10_15_1',
        'rv012_25_5_10_0', 'price'
    ]

    # 识别出需要特殊处理的列（不遵循“同整数”规则）
    special_cols = {'trade_time', 'code', 'nxt1_ret', 'price'}
    # 识别出需要应用“同整数，不同小数”规则的因子列
    factor_cols = [col for col in columns if col not in special_cols]

    # --- 2. 创建数据字典 ---
    data = {}

    # --- 3. 生成特殊列的数据 ---
    # trade_time: 生成5000个分钟级的时间戳
    data['trade_time'] = pd.to_datetime(
        pd.date_range(start='2023-01-01 09:30', periods=num_rows, freq='T'))

    # code: 为所有行设置一个或多个示例代码
    # 这里我们用 cycling 的方式填充三个不同的股票代码
    codes = ['RB']
    data['code'] = [codes[i % len(codes)] for i in range(num_rows)]

    # nxt1_ret: 生成随机的收益率（例如在-5%到+5%之间）
    data['nxt1_ret'] = np.random.uniform(-0.05, 0.05, size=num_rows)

    # price: 生成随机的价格（例如在10到100之间）
    data['price'] = np.random.uniform(10.0, 100.0, size=num_rows)

    # --- 4. 生成因子列的数据（核心逻辑） ---
    # 为了实现“每行整数位相同，小数位不同”，我们使用 NumPy 的广播功能，非常高效

    # a. 为每一行生成一个基础随机整数（例如0到50之间）
    #    形状为 (5000, 1)，以便后续广播
    base_integers = np.random.randint(0, 51, size=(num_rows, 1))

    # b. 为所有因子列生成随机小数（0到1之间）
    #    形状为 (5000, num_factor_cols)
    num_factor_cols = len(factor_cols)
    random_decimals = np.random.rand(num_rows, num_factor_cols)

    # c. 将基础整数和随机小数相加
    #    NumPy广播机制会自动将 (5000, 1) 的整数加到 (5000, N) 的小数矩阵的每一列上
    factor_values = base_integers + random_decimals

    # d. 将生成的因子数据填充到数据字典中
    for i, col_name in enumerate(factor_cols):
        data[col_name] = factor_values[:, i]

    # --- 5. 创建并整理 DataFrame ---
    # 使用数据字典创建 DataFrame
    df = pd.DataFrame(data)

    # 确保列的顺序与您提供的原始顺序完全一致
    df = df[columns]
    return df
