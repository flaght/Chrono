"""
因子代号: mc003_005
原历史名: voi_1
因子名称: 经典挂单差变动
所属分类: 一档订单流动力学 (order_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sum (BidVol1_t - BidVol1_{t-1}) - (AskVol1_t - AskVol1_{t-1})$
二次加工评级: 🟡 黄灯（绝对手数，建议使用 voi_normalized）

因子说明:
    经典委托量变动 (Volume Order Imbalance)，衡量买卖委托量增量的相对差额。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_voi").sum().alias("mc003_005")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc003_005 (voi_1): 经典挂单差变动。

    计算逻辑:
        汇总1分钟内买卖一档挂单增量之差 (VOI_t) 的累计总量。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc003_005 列
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
        .select(["trade_time", "code", "symbol", "mc003_005"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
