"""
因子代号: mc003_004
原历史名: ofi_normalized
因子名称: 成交量归一化 OFI
所属分类: 一档订单流动力学 (order_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{ofi\_sum}{volume + \epsilon}$
二次加工评级: 🟢 绿灯（真实供需推动效率，跨合约分布自洽，直接 Rolling）

因子说明:
    消除绝对成交手数量纲，衡量单位成交量所伴随的订单流真实推动效率。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    pl.when(pl.col("_delta_v").sum() > 0)
    .then(pl.col("_ofi").sum() / pl.col("_delta_v").sum())
    .otherwise(0.0)
    .alias("mc003_004")
)


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc003_004 (ofi_normalized): 成交量归一化 OFI。

    计算逻辑:
        将订单流不平衡总量除以1分钟累计成交量完成归一化。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc003_004 列
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
        .select(["trade_time", "code", "symbol", "mc003_004"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
