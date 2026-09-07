"""
因子代号: mc002_009
原历史名: realized_bipower_var
因子名称: 1分钟双幂次波动 (BV)
所属分类: 买卖盘口与微观结构类 (imbalance)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{\pi}{2}\frac{N}{N-2}\sum_{t=3}^N \|\Delta \ln P_t\| \cdot \|\Delta \ln P_{t-1}\|$
二次加工评级: 🟢 绿灯（相对比率或固定刻度，直接 Rolling 稳定）

因子说明:
    利用相邻两跳绝对收益率的乘积分离出连续扩散波动部分，具备跳跃稳健性 (Jump-Robustness)。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = ((3.141592653589793 / 2.0) * (pl.count() / (pl.count() - 2).clip(lower_bound=1)) * pl.col("_bipower_prod").sum()).alias("mc002_009")


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc002_009 (realized_bipower_var): 1分钟双幂次波动 (BV)。

    计算逻辑:
        聚合相邻两跳连续对数收益率绝对值的乘积项，并乘上连续性渐近无偏调整系数。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc002_009 列
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
        .select(["trade_time", "code", "symbol", "mc002_009"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
