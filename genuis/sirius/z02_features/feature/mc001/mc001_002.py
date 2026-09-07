"""
因子代号: mc001_002
原历史名: tick_out
因子名称: 1分钟主动卖出成交笔数
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sum \mathbb{I}_{\{D_t = -1\}}$
二次加工评级: 🟡 黄灯（绝对笔数受合约换月流动性转移影响，建议二次加工使用比率版本 tick_out_pct）

因子说明:
    统计1分钟周期内所有被判定为“主动卖出”（Seller-Initiated, D_t = -1）的 Tick 笔数。买卖方向采用改进型 Lee-Ready 规则判定。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 统计分钟内交易方向为主动卖出 (D_t = -1) 的 Tick 笔数总和
EXPR = (pl.col("_trade_dir") == -1).cast(pl.Int64).sum().alias("mc001_002")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_002 (tick_out): 1分钟主动卖出成交笔数。

    计算逻辑:
        统计1分钟内买卖方向判定为主动卖出 (D_t = -1) 的 Tick 笔数总和：sum(I_{D_t = -1})。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_002 列
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
        .select(["trade_time", "code", "symbol", "mc001_002"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
