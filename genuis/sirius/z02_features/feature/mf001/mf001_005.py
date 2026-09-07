"""
因子代号: mf001_005
原历史名: oi_volume_ratio
因子名称: 增仓成交比
所属分类: 期货持仓与博弈形态类 (open_interest)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{|delta\_oi|}{volume + \epsilon}$
二次加工评级: 🟢 绿灯（衡量沉淀资金占比，天然无量纲，直接 Rolling）

因子说明:
    衡量单边沉淀资金占总成交的份额，无量纲化特征。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    pl.when(pl.col("_delta_v").sum() > 0)
    .then((pl.col("OpenInterest").last() - pl.col("OpenInterest").first()).abs() / pl.col("_delta_v").sum())
    .otherwise(0.0)
    .alias("mf001_005")
)


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf001_005 (oi_volume_ratio): 增仓成交比。

    计算逻辑:
        将持仓净变动的绝对值除以当期累计成交量。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf001_005 列
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
        .select(["trade_time", "code", "symbol", "mf001_005"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
