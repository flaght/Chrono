"""
因子代号: mf003_003
原历史名: avg_price_slope
因子名称: 当日均价斜率
所属分类: 期货日内均价线与基准锚点类 (average_price_anchor)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\\frac{AveragePrice_{last} - AveragePrice_{first}}{AveragePrice_{first} + \\epsilon}$
二次加工评级: 🟢 绿灯（均价移动相对速率，直接 Rolling）

因子说明:
    衡量1分钟内全天官方累积加权均价的微变斜率，反映全天主力加权成本重心的边际变动方向与速率。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    (pl.col("AveragePrice").last() - pl.col("AveragePrice").first())
    / (pl.col("AveragePrice").first() + 1e-7)
).fill_nan(0.0).fill_null(0.0).alias("mf003_003")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf003_003 (avg_price_slope): 当日均价斜率。

    计算逻辑:
        计算1分钟内日均价末首差额除以初始日均价，度量均价线的微观变动斜率。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf003_003 列
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
        .select(["trade_time", "code", "symbol", "mf003_003"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
