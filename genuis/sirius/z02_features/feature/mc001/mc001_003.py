"""
因子代号: mc001_003
原历史名: net_tick_in
因子名称: 1分钟净主动买入成交笔数
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $tick\_in - tick\_out$
二次加工评级: 🟡 黄灯（绝对笔数差额受合约换月影响，建议二次加工使用比率版本 net_tick_in_pct）

因子说明:
    计算1分钟周期内主动买入笔数与主动卖出笔数的差额，反映主动交易情绪的买卖倾斜偏度。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 主动买入笔数与主动卖出笔数之差: tick_in - tick_out
EXPR = ((pl.col("_trade_dir") == 1).cast(pl.Int64).sum() - (pl.col("_trade_dir") == -1).cast(pl.Int64).sum()).alias("mc001_003")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_003 (net_tick_in): 1分钟净主动买入成交笔数。

    计算逻辑:
        计算1分钟内主动买入笔数与主动卖出笔数之差：tick_in - tick_out，反映主动交易情绪的偏度。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_003 列
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
        .select(["trade_time", "code", "symbol", "mc001_003"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
