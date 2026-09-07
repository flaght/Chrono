"""
因子代号: mc001_024
原历史名: smart_money_out_pct
因子名称: 1分钟聪明钱卖出额占比
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{smart\_money\_out}{money + \epsilon}$
二次加工评级: 🟢 绿灯（严格处于 [0, 1]，天然无量纲，可直接跨主力无脑 Rolling）

因子说明:
    聪明钱卖出金额占总成交额的比例，反映大资金的主动空头倾斜。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks
from .mc001_022 import EXPR as mc001_022_expr

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 聪明钱卖出金额占总成交额比例: smart_money_out / (money + eps)
EXPR = (pl.col("mc001_022") / (pl.col("money") + 1e-7)).alias("mc001_024")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_024 (smart_money_out_pct): 1分钟聪明钱卖出额占比。

    计算逻辑:
        计算聪明钱主动卖出金额占分钟总成交额的比例：smart_money_out / (money + 1e-7)。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_024 列
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
            mc001_022_expr
        ])
        .with_columns([EXPR])
        .select(["trade_time", "code", "symbol", "mc001_024"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
