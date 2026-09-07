"""
因子代号: mf003_002
原历史名: vwap_to_avg_bias
因子名称: 1分VWAP相对日均价偏离
所属分类: 期货日内均价线与基准锚点类 (average_price_anchor)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\\frac{vwap - AveragePrice_{last}}{AveragePrice_{last} + \\epsilon}$
二次加工评级: 🟢 绿灯（稳健相对偏离百分比，直接 Rolling）

因子说明:
    衡量1分钟VWAP（成交量加权平均价）相对于全天官方累积均价的偏离程度，平滑微观噪声，比单纯收盘价更具资金代表性。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 1分钟成交量加权均价（无成交量时以收盘价兜底）
_vwap = (
    pl.when(pl.col("_delta_v").sum() > 0)
    .then(pl.col("_vol_price_prod").sum() / (pl.col("_delta_v").sum() + 1e-7))
    .otherwise(pl.col("LastPrice").last())
)

# 因子核心计算表达式
EXPR = (
    (_vwap - pl.col("AveragePrice").last())
    / (pl.col("AveragePrice").last() + 1e-7)
).fill_nan(0.0).fill_null(0.0).alias("mf003_002")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf003_002 (vwap_to_avg_bias): 1分VWAP相对日均价偏离。

    计算逻辑:
        计算1分钟内Tick成交量加权均价 (VWAP) 相对末尾日内均价 AveragePrice 的相对偏离百分比。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf003_002 列
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
        .select(["trade_time", "code", "symbol", "mf003_002"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
