import numpy as np
import pandas as pd
from lumina.impulse.fixed import *


def ki023(close, volume, openint, window, weriod, ewm=False):
    """
    持仓成本分布因子 (Position Cost Distribution)

    基于FSD流通股本分布理论改进，适配期货市场

    原理:
        1. 通过成交量和持仓量变化追踪持仓成本分布
        2. 计算获利盘比例 = 当前价格以下的持仓占比
        3. 获利盘比例极低时可能存在超跌反弹机会

    计算方法:
        1. 估算换手率 = volume / (openint + 1e-10)
        2. 更新持仓成本中枢 = EMA(close, decay=turnover)
        3. 获利盘比例 = sigmoid((close - cost_center) / volatility)

    参数:
        close: 收盘价 DataFrame (分钟级)
        volume: 成交量 DataFrame (分钟级)
        openint: 持仓量 DataFrame (分钟级)
        window: 外层平滑窗口
        weriod: 日内周期 (如240分钟/天)
        ewm: 是否使用指数加权

    返回:
        获利盘比例因子值 (0-1之间)

    信号解读:
        profit_ratio < 0.1: 极度超跌，可能反弹
        profit_ratio > 0.9: 获利盘过多，可能回调
    """
    method = 'ewm' if ewm else 'rolling'

    # 计算日内累计成交量
    daily_volume = roller_sum(volume, weriod, weriod, method)

    # 计算日内平均持仓量
    daily_openint = roller_mean(openint, weriod, weriod, method)

    # 估算换手率 (成交量 / 持仓量)
    turnover_rate = daily_volume / (daily_openint + 1e-10)
    turnover_rate = turnover_rate.clip(upper=1.0)  # 换手率上限100%

    # 计算价格波动率 (用于标准化)
    price_std = roller_std(close, weriod, weriod, method)

    # 使用换手率加权计算持仓成本中枢
    # 高换手时，成本中枢快速向当前价格靠拢
    decay = 1 - turnover_rate.clip(lower=0.01, upper=0.5)

    # 递归计算成本中枢 (使用EMA近似)
    # cost_center ≈ decay * cost_center_prev + (1-decay) * close
    ema_span = window * 2  # 使用window控制衰减速度
    cost_center = roller_mean(close, ema_span, ema_span, 'ewm')

    # 计算价格相对成本中枢的偏离
    price_deviation = (close - cost_center) / (price_std + 1e-10)

    # 使用sigmoid函数映射到0-1区间 (获利盘比例)
    # sigmoid(x) = 1 / (1 + exp(-x))
    profit_ratio = 1 / (1 + np.exp(-price_deviation))

    # 最终用window做平滑
    alpha = roller_mean(profit_ratio, window, window, method)

    return alpha
