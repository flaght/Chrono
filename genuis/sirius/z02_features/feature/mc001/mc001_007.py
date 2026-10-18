"""
因子代号: mc001_007
原历史名: volume_in
因子名称: 1分钟主动买入成交量
所属分类: 资金流向与大单类 (money_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\sum \Delta V_t \cdot \mathbb{I}_{\{D_t = +1\}}$
二次加工评级: 🟡 黄灯（绝对手数受换月流动性转移影响，建议二次加工使用 volume_in_pct）

因子说明:
    累加1分钟内所有主动买入 Tick 的成交量增量（ΔV），度量多头主动吃单推动的绝对规模。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式（供单因子独立计算及批次聚合器 aggregator 统一引用）
# 逻辑说明: 筛选主动买入 (D_t = 1) 的增量成交量并求和
EXPR = pl.col("_delta_v").filter(pl.col("_trade_dir") == 1).sum().fill_null(0).alias("mc001_007")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc001_007 (volume_in): 1分钟主动买入成交量。

    计算逻辑:
        筛选买卖方向为主动买入 (D_t = +1) 的 Tick，累加其单跳成交量增量：sum(ΔV_t * I_{D_t = 1})。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc001_007 列
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
        .select(["trade_time", "code", "symbol", "mc001_007"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
