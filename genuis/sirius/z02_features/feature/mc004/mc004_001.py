"""
因子代号: mc004_001
原历史名: corr_money_ret
因子名称: 成交额与收益率相关性
所属分类: 微观交互与协方差类 (corr)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\mathrm{corr}(\Delta M_t, \Delta \ln LastPrice_t)$
二次加工评级: 🟢 绿灯（Pearson 相关系数严格处于 [-1, 1]，直接 Rolling）

因子说明:
    衡量成交金额与价格收益率的时序协动性。正相关代表放量上涨/缩量下跌（资金主动做多），负相关代表放量下跌（恐慌出逃）。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = pl.corr("_delta_m", "_log_ret").fill_nan(0.0).fill_null(0.0).alias("mc004_001")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc004_001 (corr_money_ret): 成交额与收益率相关性。

    计算逻辑:
        计算1分钟内Tick增量成交额与对数收益率的 Pearson 相关系数。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc004_001 列
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
        .select(["trade_time", "code", "symbol", "mc004_001"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
