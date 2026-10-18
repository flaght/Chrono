"""
因子代号: mc004_004
原历史名: corr_vwap_spread
因子名称: VWAP 与价差相关性
所属分类: 微观交互与协方差类 (corr)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\mathrm{corr}(P_{vwap\_tick, t}, Spread_t)$
二次加工评级: 🟢 绿灯（Pearson 相关系数严格处于 [-1, 1]，直接 Rolling）

因子说明:
    衡量逐跳均价走势与盘口做市宽度的关联度，反映价格处于高位/低位时的流动性松紧。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.corr("_tick_vwap", "_spread").fill_nan(0.0).fill_null(0.0).alias("mc004_004")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc004_004 (corr_vwap_spread): VWAP 与价差相关性。

    计算逻辑:
        计算1分钟内逐跳微观VWAP均价与买卖价差的 Pearson 相关系数。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc004_004 列
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
        .select(["trade_time", "code", "symbol", "mc004_004"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
