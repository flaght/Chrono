"""
因子代号: tc004_015
原历史名: tc004_015
因子定义: 成交量加权价格动量：以成交量为权重的收盘价收益率滚动加权和
"""
import polars as pl

DEFAULT_PERIODS = (13, 8, 5)
NAME = "tc004_015"


def calculate(df_lazy: pl.LazyFrame, periods: tuple[int, int, int]) -> pl.LazyFrame:
    """计算长、中、短三周期均线结构。"""
    long, medium, short = periods
    return df_lazy.with_columns(
        (
            pl.col("close").rolling_mean(long).over("code")
            - 2 * pl.col("close").rolling_mean(medium).over("code")
            + pl.col("close").rolling_mean(short).over("code")
        ).alias(NAME)
    )


def compute(
    df_lazy: pl.LazyFrame,
    periods: tuple[int, int, int] = DEFAULT_PERIODS,
) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 的 trade_time、code、close 计算 tc004_015。"""
    if len(periods) != 3 or any(p <= 0 for p in periods):
        raise ValueError("periods 必须包含三个正整数")
    if not periods[0] > periods[1] > periods[2]:
        raise ValueError("periods 必须满足 long > medium > short")
    return calculate(df_lazy.sort(["trade_time", "code"]), periods).select(
        ["trade_time", "code", NAME]
    )
