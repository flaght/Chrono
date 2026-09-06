"""
因子代号: mc002_007
原历史名: micro_price_bias
因子名称: 1分钟微观价偏离均值
所属分类: 买卖盘口与微观结构类 (imbalance)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{1}{N}\sum \frac{P_{micro, t} - P_{mid, t}}{P_{mid, t}}$
二次加工评级: 🟢 绿灯（相对比率或固定刻度，直接 Rolling 稳定）

因子说明:
    微观价格 (Micro-price) 将买卖挂单量作为权重对盘口做加权平均。本因子衡量微观价格对中价的相对偏离均值，反映盘口即时价格重心的隐性倾向。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_micro_bias").mean().alias("mc002_007")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc002_007 (micro_price_bias): 1分钟微观价偏离均值。

    计算逻辑:
        统计1分钟内微观价格相对中间价的相对偏离百分比均值。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc002_007 列
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
        .select(["trade_time", "code", "symbol", "mc002_007"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
