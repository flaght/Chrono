"""
因子代号: mc003_007
原历史名: bid_depletion_rate
因子名称: 买单消耗/撤退速度
所属分类: 一档订单流动力学 (order_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sum \mathbb{I}_{\{BidPrice1_t < BidPrice1_{t-1}\}} \cdot BidVolume1_{t-1}$
二次加工评级: 🟡 黄灯（建议使用 bid_depletion_ratio）

因子说明:
    买方一档价格下移时原买单的被动吞噬或主动撤单总量。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_bid_depletion").sum().alias("mc003_007")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc003_007 (bid_depletion_rate): 买单消耗/撤退速度。

    计算逻辑:
        汇总1分钟内买方一档挂单被向下击穿或撤离的总手数。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc003_007 列
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
        .select(["trade_time", "code", "symbol", "mc003_007"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
