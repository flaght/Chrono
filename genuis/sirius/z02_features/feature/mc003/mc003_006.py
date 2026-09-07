"""
因子代号: mc003_006
原历史名: voi_normalized
因子名称: 成交量归一化 VOI
所属分类: 一档订单流动力学 (order_flow)
数据粒度: CTP Level-1 Tick 快照流 -> 1分钟下采样特征
计算公式: $\frac{voi\_1}{volume + \epsilon}$
二次加工评级: 🟢 绿灯（消除绝对成交手数量纲，直接 Rolling）

因子说明:
    消除绝对成交手数量纲，衡量买卖一档挂单净调整对成交量的相对占比。
"""

from __future__ import annotations
import polars as pl
from feature.utils.preprocess import preprocess_ticks

# 因子核心计算表达式
EXPR = (
    pl.when(pl.col("_delta_v").sum() > 0)
    .then(pl.col("_voi").sum() / pl.col("_delta_v").sum())
    .otherwise(0.0)
    .alias("mc003_006")
)


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算微观结构因子 mc003_006 (voi_normalized): 成交量归一化 VOI。

    计算逻辑:
        将经典挂单差变动总量除以1分钟累计成交量完成归一化。

    参数:
        df_lazy: pl.LazyFrame，输入 Tick 级别数据流
    返回:
        pl.LazyFrame: 包含 trade_time, code, symbol, mc003_006 列
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
        .select(["trade_time", "code", "symbol", "mc003_006"])
        .sort(["trade_time", "code", "symbol"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
