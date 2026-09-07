"""
因子代号: mc004_005
原历史名: corr_vol_depth_imb
因子名称: 成交量与挂单不平衡相关性
所属分类: 微观交互与协方差类 (corr)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\mathrm{corr}(\Delta V_t, \text{depth\_imbalance}_t)$
二次加工评级: 🟢 绿灯（Pearson 相关系数严格处于 [-1, 1]，直接 Rolling）

因子说明:
    衡量成交放量时盘口挂单偏向，正值表明成交放量多发生在买盘深度堆积阶段（积极吃单）。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.corr("_delta_v", "_depth_imb").fill_nan(0.0).fill_null(0.0).alias("mc004_005")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc004_005 (corr_vol_depth_imb): 成交量与挂单不平衡相关性。

    计算逻辑:
        计算1分钟内Tick增量成交量与盘口挂单深度不平衡比率的 Pearson 相关系数。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc004_005 列
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
        .select(["trade_time", "code", "symbol", "mc004_005"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
