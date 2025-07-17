import pandas as pd

def equal_weight_synthesis(positions: pd.DataFrame):
    # 求和，得到每个时间点的净信号强度
    net_positions = positions.sum(axis=1)
    # 归一化：将信号强度缩放到[-1, 1]区间
    # 除以子策略的总数，就得到了平均信号强度。
    # 将结果控制在[-1, 1]内
    meta_positions = net_positions / len(positions.columns)
    meta_positions.name = 'equal_weight'
    return meta_positions


def fitness_weight_synthesis(positions: pd.DataFrame, programs: pd.DataFrame,
                             fitness_name: str):
    programs = programs.set_index('name')
    weights = programs.loc[positions.columns, fitness_name]
    ## 处理负数情况(正常情况不会被选中)
    weights[weights < 0] = 0

    total_weight = weights.sum()
    normalized_weights = weights / total_weight
    meta_positions = positions.mul(normalized_weights,
                                   axis='columns').sum(axis=1)
    meta_positions.name = f'{fitness_name}_weight'
    return meta_positions


def volatility_weight_synthesis(positions: pd.DataFrame):
    # 1. 计算每个策略信号的波动率（日均换手强度）
    # diff()计算每日信号变化，abs()取绝对值，mean()求平均
    volatilities = positions.diff().abs().mean()

    # 2. 计算倒数权重
    # +1e-8是为了防止除以零（对于恒定信号）
    inverse_volatilities = 1 / (volatilities + 1e-8)

    # 3. 归一化权重
    total_inverse_vol = inverse_volatilities.sum()
    normalized_weights = inverse_volatilities / total_inverse_vol

    meta_positions = positions.mul(normalized_weights,
                                   axis='columns').sum(axis=1)

    meta_positions.name = 'vol_inv_weight'
    return meta_positions
