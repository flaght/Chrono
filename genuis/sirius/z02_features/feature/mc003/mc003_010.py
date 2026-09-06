"""
因子代号: mc003_010
原历史名: ask_depletion_ratio
因子名称: 卖单防线击穿比率
所属分类: 一档订单流动力学 (order_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{ask\_depletion\_rate}{volume + \epsilon}$
二次加工评级: 🟢 绿灯（相对成交量占比，直接 Rolling）

因子说明:
    卖单防线溃败量占当期总成交量的比率，衡量多头突破卖方防线的效率。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (pl.col("_ask_depletion").sum() / (pl.col("_delta_v").sum() + 1e-7)).alias("mc003_010")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc003_010 (ask_depletion_ratio): 卖单防线击穿比率。

    计算逻辑:
        将卖单防线击穿总量除以当期累计成交量进行归一化。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc003_010 列
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
        .select(["trade_time", "code", "symbol", "mc003_010"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
