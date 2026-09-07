"""
因子代号: mc002_004
原历史名: depth_imbalance_1
因子名称: 1分钟挂单深度不平衡均值
所属分类: 买卖盘口与微观结构类 (imbalance)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{1}{N}\sum \frac{BidVolume1_t - AskVolume1_t}{BidVolume1_t + AskVolume1_t}$
二次加工评级: 🟢 绿灯（相对比率或固定刻度，直接 Rolling 稳定）

因子说明:
    统计买卖一档委托量的相对失衡比率，值域 [-1, 1]，正值代表买盘委托显著占优，反映超高频即时供给/需求倾斜。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_depth_imb").mean().alias("mc002_004")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc002_004 (depth_imbalance_1): 1分钟挂单深度不平衡均值。

    计算逻辑:
        统计1分钟内买卖一档委托挂单深度不平衡比率的算术均值。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc002_004 列
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
        .select(["trade_time", "code", "symbol", "mc002_004"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
