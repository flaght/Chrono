"""
因子代号: mc002_006
原历史名: depth_imbalance_last
因子名称: 1分钟尾部挂单不平衡
所属分类: 买卖盘口与微观结构类 (imbalance)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: 分钟末尾 Tick 的 $\frac{BidVolume1 - AskVolume1}{BidVolume1 + AskVolume1}$
二次加工评级: 🟢 绿灯（相对比率或固定刻度，直接 Rolling 稳定）

因子说明:
    提取1分钟最后时刻的瞬时深度失衡，表征该分钟收盘瞬间多空盘口意愿偏向，对下一分钟首跳具备强烈即时动量先导性。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_depth_imb").last().alias("mc002_006")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc002_006 (depth_imbalance_last): 1分钟尾部挂单不平衡。

    计算逻辑:
        提取1分钟K线结束时刻（末跳 Tick）的买卖一档深度不平衡比率。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc002_006 列
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
        .select(["trade_time", "code", "symbol", "mc002_006"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
