"""
因子代号: mc002_010
原历史名: jump_ratio
因子名称: 1分钟价格跳跃波动占比
所属分类: 买卖盘口与微观结构类 (imbalance)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\max\left(0, \frac{RV^2 - BV}{RV^2 + \epsilon}\right)$
二次加工评级: 🟢 绿灯（相对比率或固定刻度，直接 Rolling 稳定）

因子说明:
    通过 Barner-Nielsen & Shephard 双幂次分解，计算跳跃成分在总已实现方差中的相对占比，严格处于 [0, 1]。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.max_horizontal(pl.lit(0.0), (pl.col("mc002_008").pow(2) - pl.col("mc002_009")) / (pl.col("mc002_008").pow(2) + 1e-7)).clip(0.0, 1.0).alias("mc002_010")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc002_010 (jump_ratio): 1分钟价格跳跃波动占比。

    计算逻辑:
        根据 BNS 跳跃分解测度，计算总已实现方差中剔除连续扩散波动后的非连续跳跃部分占比。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc002_010 列
    """
    primitives = preprocess_ticks(df_lazy)
    base = (
        primitives
        .with_columns([
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
        ])
        .group_by(["trade_time", "code", "symbol"])
    )
    from .mc002_008 import EXPR as _rv_expr
    from .mc002_009 import EXPR as _bv_expr
    return (
        base.agg([_rv_expr, _bv_expr])
        .with_columns([EXPR])
        .select(["trade_time", "code", "symbol", "mc002_010"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
