"""
因子代号: mf001_011
原历史名: swap_volume_ratio
因子名称: 换手成交量占比
所属分类: 期货持仓与博弈形态类 (open_interest)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{swap\_volume}{volume + \epsilon}$
二次加工评级: 🟢 绿灯（严格处于 [0, 1]，筹码纯换手率，直接 Rolling）

因子说明:
    换手量在总成交量中的比率，严格处于 [0, 1]，反映筹码纯换手率。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    pl.when(pl.col("_delta_v").sum() > 0)
    .then(pl.col("_swap_vol").sum() / pl.col("_delta_v").sum())
    .otherwise(0.0)
    .clip(0.0, 1.0)
    .alias("mf001_011")
)


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf001_011 (swap_volume_ratio): 换手成交量占比。

    计算逻辑:
        将换手成交量除以总成交量进行归一化。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf001_011 列
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
        .select(["trade_time", "code", "symbol", "mf001_011"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
