"""
因子代号: mf002_003
原历史名: limit_bound_asymmetry
因子名称: 涨跌停距离不对称性
所属分类: 期货涨跌停极限边界与流动性挤压类 (limit_bounds)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\\frac{dist\\_upper\\_limit - dist\\_lower\\_limit}{dist\\_upper\\_limit + dist\\_lower\\_limit}$
二次加工评级: 🟢 绿灯（严格处于 $[-1, 1]$，直接 Rolling）

因子说明:
    衡量价格相对涨停板和跌停板的空间不对称度。正值表示距涨停空间大（价格偏向跌停侧），负值表示距跌停空间大（价格偏向涨停侧）。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

_dist_up = (pl.col("UpperLimitPrice").last() - pl.col("LastPrice").last()) / (pl.col("LastPrice").last() + 1e-7)
_dist_down = (pl.col("LastPrice").last() - pl.col("LowerLimitPrice").last()) / (pl.col("LastPrice").last() + 1e-7)

# 因子核心计算表达式
EXPR = (
    (_dist_up - _dist_down) / (_dist_up + _dist_down + 1e-7)
).clip(-1.0, 1.0).fill_nan(0.0).fill_null(0.0).alias("mf002_003")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mf002_003 (limit_bound_asymmetry): 涨跌停距离不对称性。

    计算逻辑:
        计算1分钟末尾距涨停空间与跌停空间的相对差比，度量多空极限空间的偏态分布。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mf002_003 列
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
        .select(["trade_time", "code", "symbol", "mf002_003"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
