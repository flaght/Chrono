"""
因子代号: mf001_020
原历史名: oi_flow_imbalance
因子名称: 增仓能量不平衡
所属分类: 期货持仓与博弈形态类 (open_interest)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{bull\_active\_open - bear\_active\_open}{volume + \epsilon}$
二次加工评级: 🟢 绿灯（严格处于 [-1, 1]，天然多空能量比，直接 Rolling）

因子说明:
    主动多头增仓与主动空头增仓的净差额比率，严格处于 [-1, 1]，天然反映多空主动建仓能量差。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    pl.when(pl.col("_delta_v").sum() > 0)
    .then((pl.col("_bull_active_open").sum() - pl.col("_bear_active_open").sum()) / pl.col("_delta_v").sum())
    .otherwise(0.0)
    .clip(-1.0, 1.0)
    .alias("mf001_020")
)


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf001_020 (oi_flow_imbalance): 增仓能量不平衡。

    计算逻辑:
        计算主动多头增仓量与主动空头增仓量之差，除以总成交量完成归一化。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf001_020 列
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
        .select(["trade_time", "code", "symbol", "mf001_020"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
