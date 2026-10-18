"""
因子代号: mc001_018
原历史名: smart_volume_out
因子名称: 1分钟聪明钱卖出量
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sum \Delta V_t \cdot \mathbb{I}_{\{D_t = -1 \text{ 且 } \Delta V_t \ge Q_{90}(\Delta V)\}}$
二次加工评级: 🟡 黄灯（建议二次加工使用 smart_volume_out_pct）

因子说明:
    筛选成交量超过 90% 分位数（大单）的主动卖出 Tick 成交量之和，捕捉机构与大单空头离场行为。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 大单分位数阈值筛选: D_t=-1 且 ΔV_t >= quantile(0.9) 的成交量求和
EXPR = pl.col("_delta_v").filter((pl.col("_trade_dir") == -1) & (pl.col("_delta_v") >= pl.col("_delta_v").quantile(0.90))).sum().fill_null(0).alias("mc001_018")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_018 (smart_volume_out): 1分钟聪明钱卖出量。

    计算逻辑:
        筛选主动卖出 (D_t = -1) 且单跳增量 ΔV_t 达到或超过分钟内 90% 分位数的大单成交量之和。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_018 列
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
        .select(["trade_time", "code", "symbol", "mc001_018"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
