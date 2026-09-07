"""
因子代号: mc002_005
原历史名: depth_imbalance_std
因子名称: 1分钟挂单不平衡波动率
所属分类: 买卖盘口与微观结构类 (imbalance)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\mathrm{std}\left(\frac{BidVolume1_t - AskVolume1_t}{BidVolume1_t + AskVolume1_t}\right)$
二次加工评级: 🟢 绿灯（相对比率或固定刻度，直接 Rolling 稳定）

因子说明:
    统计挂单深度失衡的时序样本离散度，反映盘口买卖挂单多空博弈力量的反复拉锯与流动性稳定性。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_depth_imb").std().fill_null(0.0).alias("mc002_005")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc002_005 (depth_imbalance_std): 1分钟挂单不平衡波动率。

    计算逻辑:
        统计1分钟内挂单深度不平衡比率的时序样本标准差。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc002_005 列
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
        .select(["trade_time", "code", "symbol", "mc002_005"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
