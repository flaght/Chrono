"""
因子代号: tc006_037
原历史名: tc006_037
因子定义: 20周期衰减加权均线与 SMA 偏离 DECAYLINEAR(close, 20) / SMA(close, 20) - 1
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 20 周期衰减加权均线与 SMA 偏离 MOMENTUM_DECAY。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_037 列
    """
    shifts = [pl.col('close').shift(i).over('code').alias(f'_c_{i}') for i in range(20)]

    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(shifts)
        .with_columns(
            (sum((20 - i) * pl.col(f'_c_{i}') for i in range(20)) / 210.0).alias('_decay_close'),
            pl.col('close').rolling_mean(20).over('code').alias('_sma20'),
        )
        .with_columns(
            (safe_div(pl.col('_decay_close'), pl.col('_sma20')) - 1.0).alias("tc006_037")
        )
        .select(['trade_time', 'code', "tc006_037"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_037；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
