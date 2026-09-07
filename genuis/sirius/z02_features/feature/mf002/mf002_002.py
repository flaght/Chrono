"""
因子代号: mf002_002
原历史名: dist_lower_limit
因子名称: 距离跌停板相对幅度
所属分类: 期货涨跌停极限边界与流动性挤压类 (limit_bounds)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\\frac{close - LowerLimitPrice}{close}$
二次加工评级: 🟢 绿灯（相对百分比空间，直接 Rolling）

因子说明:
    衡量1分钟末尾收盘价距离当日交易所法定跌停板价位的相对百分比下行空间。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    (pl.col("LastPrice").last() - pl.col("LowerLimitPrice").last())
    / (pl.col("LastPrice").last() + 1e-7)
).fill_nan(0.0).fill_null(0.0).alias("mf002_002")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf002_002 (dist_lower_limit): 距离跌停板相对幅度。

    计算逻辑:
        计算1分钟末尾 LastPrice 相对 LowerLimitPrice 的相对价差百分比。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf002_002 列
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
        .select(["trade_time", "code", "symbol", "mf002_002"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
