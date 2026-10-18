"""
因子代号: mc001_016
原历史名: net_money_in_pct
因子名称: 1分钟净流入金额占比
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{net\_money\_in}{money + \epsilon}$
二次加工评级: 🟢 绿灯（严格处于 [-1, 1]，天然无量纲，可直接跨主力无脑 Rolling）

因子说明:
    净流入金额占分钟总成交额的比例，归一化消除金额量纲，反映主力资金的净推动效率。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks
from .mc001_013 import EXPR as mc001_013_expr
from .mc001_014 import EXPR as mc001_014_expr

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 净流入金额占总成交额比例: (money_in - money_out) / (money + eps)
EXPR = ((pl.col("mc001_013").cast(pl.Float64) - pl.col("mc001_014").cast(pl.Float64)) / (pl.col("money").cast(pl.Float64) + 1e-7)).alias("mc001_016")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_016 (net_money_in_pct): 1分钟净流入金额占比。

    计算逻辑:
        计算1分钟内净流入金额占分钟总成交额的比例：(money_in - money_out) / (money + 1e-7)。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_016 列
    """
    primitives = preprocess_ticks(df_lazy)
    return (
        primitives
        .with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
        ])
        .group_by(["trade_time", "code", "symbol"])
        .agg([
            pl.col("_delta_m").sum().alias("money"),
            mc001_013_expr,
            mc001_014_expr
        ])
        .with_columns([EXPR])
        .select(["trade_time", "code", "symbol", "mc001_016"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
