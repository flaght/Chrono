"""
因子代号: mc003_002
原历史名: ofi_mean
因子名称: OFI 每跳均值
所属分类: 一档订单流动力学 (order_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{1}{N}\sum_{t=1}^N OFI_t$
二次加工评级: 🟡 黄灯（绝对单跳手数，受流动性转移影响）

因子说明:
    衡量单跳平均真实供需推动强度。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.col("_ofi").mean().alias("mc003_002")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc003_002 (ofi_mean): OFI 每跳均值。

    计算逻辑:
        统计1分钟内订单流不平衡 (OFI_t) 的算术均值。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc003_002 列
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
        .select(["trade_time", "code", "symbol", "mc003_002"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
