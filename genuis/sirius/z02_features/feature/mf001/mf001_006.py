"""
因子代号: mf001_006
原历史名: double_open_vol
因子名称: 双开成交量
所属分类: 期货持仓与博弈形态类 (open_interest)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sum \Delta V_t \cdot \mathbb{I}_{\{\Delta OI_t > 0 \text{ 且 } \Delta OI_t \ge 0.8 \Delta V_t\}}$
二次加工评级: 🟡 黄灯（绝对手数，建议使用 double_open_ratio）

因子说明:
    捕捉新多头与新空头同时开仓入场的激烈对抗成交量。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_double_open_vol").sum().alias("mf001_006")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf001_006 (double_open_vol): 双开成交量。

    计算逻辑:
        汇总满足双开判定条件 (\Delta OI \ge 0.8 \Delta V 且 \Delta OI > 0) 的成交量之和。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf001_006 列
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
        .select(["trade_time", "code", "symbol", "mf001_006"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
