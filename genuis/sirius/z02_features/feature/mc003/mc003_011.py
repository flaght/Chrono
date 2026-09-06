"""
因子代号: mc003_011
原历史名: depletion_imbalance
因子名称: 挂单防线失守差比
所属分类: 一档订单流动力学 (order_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{bid\_depletion\_rate - ask\_depletion\_rate}{volume + \epsilon}$
二次加工评级: 🟢 绿灯（严格处于 [-1, 1]，买空防线溃败差，直接 Rolling）

因子说明:
    多空挂单防线失守差额除以总成交量，严格处于 [-1, 1]，正值代表买盘防线溃败更严重（空头更强势）。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    pl.when(pl.col("_delta_v").sum() > 0)
    .then((pl.col("_bid_depletion").sum() - pl.col("_ask_depletion").sum()) / pl.col("_delta_v").sum())
    .otherwise(0.0)
    .clip(-1.0, 1.0)
    .alias("mc003_011")
)


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc003_011 (depletion_imbalance): 挂单防线失守差比。

    计算逻辑:
        计算买单防线消耗量与卖单防线消耗量之差，并除以总成交量完成归一化。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc003_011 列
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
        .select(["trade_time", "code", "symbol", "mc003_011"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
