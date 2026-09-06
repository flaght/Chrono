"""
因子代号: mf002_004
原历史名: near_limit_liquidity
因子名称: 极限位置流动性比率
所属分类: 期货涨跌停极限边界与流动性挤压类 (limit_bounds)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: 当 $\\min(dist\\_up, dist\\_low) \\le 0.005$ 时的 $\\frac{BidVol1 + AskVol1}{\\Delta V_t + \\epsilon}$
二次加工评级: 🟢 绿灯（挂单深度与成交相对比，直接 Rolling）

因子说明:
    监控最新成交价贴近涨跌停极限位置（距板价小于千分之五）时的挂单深度相对每跳成交量的比率，反映逼近极限边界时的流动性衰竭与承接厚度。若无逼近状态则为 0。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式（贴近板价时的平均流动性比率，若无触发则为 0.0）
EXPR = pl.col("_near_limit_liq").mean().fill_nan(0.0).fill_null(0.0).alias("mf002_004")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf002_004 (near_limit_liquidity): 极限位置流动性比率。

    计算逻辑:
        计算1分钟内处于涨跌停边缘（距板幅度 <= 0.5%）的 Tick 盘口挂单深度相对逐跳成交量的平均比率。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf002_004 列
    """
    primitives = preprocess_ticks(df_lazy)
    return (
        primitives
        .with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
        ])
        .group_by(["trade_time", "code", "symbol"])
        .agg([
            EXPR
        ])
        .select(["trade_time", "code", "symbol", "mf002_004"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
