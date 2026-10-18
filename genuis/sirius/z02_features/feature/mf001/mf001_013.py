"""
因子代号: mf001_013
原历史名: bear_active_open
因子名称: 主动空头增仓量
所属分类: 期货持仓与博弈形态类 (open_interest)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sum \Delta V_t \cdot \mathbb{I}_{\{D_t = -1 \text{ 且 } \Delta OI_t > 0\}}$
二次加工评级: 🟡 黄灯（建议使用 bear_active_open_ratio）

因子说明:
    主动卖出且伴随增仓，代表空头主力压制建仓量。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_bear_active_open").sum().alias("mf001_013")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf001_013 (bear_active_open): 主动空头增仓量。

    计算逻辑:
        汇总主动卖出驱动且伴随增仓的成交手数。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf001_013 列
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
        .select(["trade_time", "code", "symbol", "mf001_013"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
