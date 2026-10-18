"""
因子代号: mc002_008
原历史名: realized_volatility
因子名称: 1分钟已实现波动率 (RV)
所属分类: 买卖盘口与微观结构类 (imbalance)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sqrt{\sum_{t=2}^N \left(\ln \frac{LastPrice_t}{LastPrice_{t-1}}\right)^2}$
二次加工评级: 🟢 绿灯（相对比率或固定刻度，直接 Rolling 稳定）

因子说明:
    聚合 Tick 间对数收益率二阶矩开平方，捕捉日内超高频连续与跳跃波动的全量已实现离散度。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_sq_log_ret").sum().sqrt().alias("mc002_008")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc002_008 (realized_volatility): 1分钟已实现波动率 (RV)。

    计算逻辑:
        汇总1分钟内Tick间连续对数收益率的二阶矩（平方和），开平方得到高频已实现波动率 RV。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc002_008 列
    """
    primitives = preprocess_ticks(df_lazy)
    base = (
        primitives
        .with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
        ])
        .group_by(["trade_time", "code", "symbol"])
    )
    return (
        base.agg([EXPR])
        .select(["trade_time", "code", "symbol", "mc002_008"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
