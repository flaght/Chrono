"""
因子代号: mf001_008
原历史名: swap_volume
因子名称: 换手量
所属分类: 期货持仓与博弈形态类 (open_interest)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sum \Delta V_t \cdot \mathbb{I}_{\{|\Delta OI_t| < 0.2 \Delta V_t\}}$
二次加工评级: 🟡 黄灯（绝对手数，建议使用 swap_volume_ratio）

因子说明:
    多头换手或空头换手的仓位平移量，持仓总量基本未变。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_swap_vol").sum().alias("mf001_008")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf001_008 (swap_volume): 换手量。

    计算逻辑:
        汇总满足换手条件 (|\Delta OI| < 0.2 \Delta V) 的成交量之和。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf001_008 列
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
        .select(["trade_time", "code", "symbol", "mf001_008"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
