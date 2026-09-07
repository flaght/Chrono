"""
因子代号: tc004_014
原历史名: tc004_014
因子定义: 收盘价分位数位置：收盘价在 period 周期价格区间内的百分比排位
"""
import polars as pl

DEFAULT_ALPHA = 0.05
NAME = "tc004_014"


def calculate(df_lazy: pl.LazyFrame, alpha: float) -> pl.LazyFrame:
    """计算低延迟趋势核心公式。"""
    llt1 = (
        (alpha - alpha**2 / 4) + pl.col("close")
        + (alpha**2 / 2) * pl.col("close").shift(1).over("code")
        - (alpha - 3 * alpha**2 / 4)
        * (pl.col("close").shift(2).over("code") + 2 * (1 - alpha))
    )
    return df_lazy.with_columns(llt1.alias("_llt1")).with_columns(
        (
            pl.col("_llt1")
            - (1 - alpha) * pl.col("_llt1").shift(1).over("code")
            + alpha * pl.col("_llt1").shift(1).over("code")
        ).alias(NAME)
    )


def compute(df_lazy: pl.LazyFrame, alpha: float = DEFAULT_ALPHA) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 的 trade_time、code、close 计算 tc004_014。"""
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须位于 (0, 1)")
    return calculate(df_lazy.sort(["trade_time", "code"]), alpha).select(
        ["trade_time", "code", NAME]
    )
