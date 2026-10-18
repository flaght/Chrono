"""
因子代号: tc006_002
原历史名: tc006_002
因子定义: 经典双均线偏离度 SMA(close, 10) / SMA(close, 30) - 1
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 SMA10 与 SMA30 比率偏离度因子。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_002 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').rolling_mean(10).over('code').alias('_sma10'),
            pl.col('close').rolling_mean(30).over('code').alias('_sma30'),
        )
        .with_columns(
            (safe_div(pl.col('_sma10'), pl.col('_sma30')) - 1.0).alias("tc006_002")
        )
        .select(['trade_time', 'code', "tc006_002"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_002；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
