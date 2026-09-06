"""
因子代号: mf003_004
原历史名: price_above_avg_time
因子名称: 均线上方时间占比
所属分类: 期货日内均价线与基准锚点类 (average_price_anchor)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\\frac{1}{N}\\sum \\mathbb{I}_{\\{LastPrice_t \\ge AveragePrice_t\\}}$
二次加工评级: 🟢 绿灯（严格处于 $[0, 1]$，时间占比，直接 Rolling）

因子说明:
    衡量1分钟内最新成交价位于日均线上方的 Tick 时间点比例，反映日内多空力量在均价线博弈时的微观压制与主导情况。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_above_avg").mean().fill_nan(0.0).fill_null(0.0).alias("mf003_004")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf003_004 (price_above_avg_time): 均线上方时间占比。

    计算逻辑:
        计算1分钟内 LastPrice >= AveragePrice 的 Tick 点数占总有效 Tick 点数的比例。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf003_004 列
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
        .select(["trade_time", "code", "symbol", "mf003_004"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
