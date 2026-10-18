"""
因子代号: mc001_006
原历史名: net_tick_in_pct
因子名称: 1分钟净买入笔数占比
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{net\_tick\_in}{tick\_count + \epsilon}$
二次加工评级: 🟢 绿灯（严格处于 [-1, 1]，天然无量纲，可直接跨主力无脑 Rolling）

因子说明:
    计算净买入笔数占总 Tick 点数的百分比，正值表示买方主导，负值表示卖方主导，具有高度跨期统计稳定性。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks
from .mc001_001 import EXPR as mc001_001_expr
from .mc001_002 import EXPR as mc001_002_expr

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 净买入笔数占分钟总采样点数比例: (tick_in - tick_out) / (tick_count + eps)
EXPR = ((pl.col("mc001_001").cast(pl.Float64) - pl.col("mc001_002").cast(pl.Float64)) / (pl.col("tick_count").cast(pl.Float64) + 1e-7)).alias("mc001_006")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_006 (net_tick_in_pct): 1分钟净买入笔数占比。

    计算逻辑:
        计算1分钟内净主动买入笔数占总采样点数的比例：(tick_in - tick_out) / (tick_count + 1e-7)。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_006 列
    """
    primitives = preprocess_ticks(df_lazy)
    return (
        primitives
        .with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
        ])
        .group_by(["trade_time", "code", "symbol"])
        .agg([
            pl.len().alias("tick_count"),
            mc001_001_expr,
            mc001_002_expr
        ])
        .with_columns([EXPR])
        .select(["trade_time", "code", "symbol", "mc001_006"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
