"""
因子代号: mf003_001
原历史名: price_to_avg_dev
因子名称: 收盘相对日均价偏离度
所属分类: 期货日内均价线与基准锚点类 (average_price_anchor)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\\frac{close - AveragePrice_{last}}{AveragePrice_{last} + \\epsilon}$
二次加工评级: 🟢 绿灯（相对偏离百分比，各合约自洽，直接 Rolling）

因子说明:
    衡量1分钟K线收盘价相对于日内官方累积加权均价的百分比偏离程度。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    (pl.col("LastPrice").last() - pl.col("AveragePrice").last())
    / (pl.col("AveragePrice").last() + 1e-7)
).fill_nan(0.0).fill_null(0.0).alias("mf003_001")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf003_001 (price_to_avg_dev): 收盘相对日均价偏离度。

    计算逻辑:
        计算1分钟末尾 LastPrice 相对 AveragePrice 的相对涨跌偏离百分比。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf003_001 列
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
        .select(["trade_time", "code", "symbol", "mf003_001"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
