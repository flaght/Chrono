"""
因子代号: mc001_010
原历史名: volume_in_pct
因子名称: 1分钟主动买入成交量占比
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{volume\_in}{volume + \epsilon}$
二次加工评级: 🟢 绿灯（严格处于 [0, 1]，天然无量纲，可直接跨主力无脑 Rolling）

因子说明:
    主动买入成交量占分钟总成交量的比例，度量多头资金推动的市场份额。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks
from .mc001_007 import EXPR as mc001_007_expr

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 主动买入成交量占总成交量比例: volume_in / (volume + eps)
EXPR = (pl.col("mc001_007") / (pl.col("volume") + 1e-7)).alias("mc001_010")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_010 (volume_in_pct): 1分钟主动买入成交量占比。

    计算逻辑:
        计算1分钟内主动买入成交量占分钟总成交量的百分比：volume_in / (volume + 1e-7)。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_010 列
    """
    primitives = preprocess_ticks(df_lazy)
    return (
        primitives
        .with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
        ])
        .group_by(["trade_time", "code", "symbol"])
        .agg([
            pl.col("_delta_v").sum().alias("volume"),
            mc001_007_expr
        ])
        .with_columns([EXPR])
        .select(["trade_time", "code", "symbol", "mc001_010"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
