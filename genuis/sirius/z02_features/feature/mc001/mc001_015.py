"""
因子代号: mc001_015
原历史名: net_money_in
因子名称: 1分钟净主动流入金额
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $money\_in - money\_out$
二次加工评级: 🟡 黄灯（建议二次加工使用 net_money_in_pct）

因子说明:
    主动买入成交金额与主动卖出成交金额的差额，刻画分钟级别净资金博弈方向。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 净主动流入金额: money_in - money_out
EXPR = (pl.col("_delta_m").filter(pl.col("_trade_dir") == 1).sum().fill_null(0.0) - pl.col("_delta_m").filter(pl.col("_trade_dir") == -1).sum().fill_null(0.0)).alias("mc001_015")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_015 (net_money_in): 1分钟净主动流入金额。

    计算逻辑:
        计算1分钟内主动买入成交金额与主动卖出成交金额之差：money_in - money_out。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_015 列
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
        .select(["trade_time", "code", "symbol", "mc001_015"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
