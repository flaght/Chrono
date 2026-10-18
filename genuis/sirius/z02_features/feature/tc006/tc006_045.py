"""
因子代号: tc006_045
原历史名: tc006_045
因子定义: 14周期加权移动平均偏离度 (close - WMA(close, 14)) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 14 周期加权移动平均偏离度 WMA14_DEV。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_045 列
    """
    shifts = [pl.col('close').shift(i).over('code').alias(f'_c_{i}') for i in range(14)]

    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(shifts)
        .with_columns(
            (sum((14 - i) * pl.col(f'_c_{i}') for i in range(14)) / 105.0).alias('_wma14')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_wma14'), pl.col('close')).alias("tc006_045")
        )
        .select(['trade_time', 'code', "tc006_045"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_045；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
