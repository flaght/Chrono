"""
因子代号: mf001_001
原历史名: delta_oi
因子名称: 持仓量净变动
所属分类: 期货持仓与博弈形态类 (open_interest)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: {last} - OpenInterest_{first}$
二次加工评级: 🟡 黄灯（净增绝对量，建议使用 delta_oi_ratio）

因子说明:
    衡量1分钟内总持仓净增减量。正增仓代表沉淀资金进场，负减仓代表资金撤离获利了结。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (pl.col("OpenInterest").last() - pl.col("OpenInterest").first()).alias("mf001_001")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf001_001 (delta_oi): 持仓量净变动。

    计算逻辑:
        计算1分钟K线末尾持仓量与起始持仓量之差。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf001_001 列
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
        .select(["trade_time", "code", "symbol", "mf001_001"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
