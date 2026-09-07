"""
因子代号: tc006_048
原历史名: tc006_048
因子定义: 归一化顺势指标 CCI(high, low, close, 20) / 100
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算归一化 20 周期顺势指标 CCI_NORM。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_048 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0).alias('_tp')
        )
        .with_columns(
            pl.col('_tp').rolling_mean(20).over('code').alias('_sma_tp')
        )
        .with_columns(
            (pl.col('_tp') - pl.col('_sma_tp')).abs().alias('_dev')
        )
        .with_columns(
            pl.col('_dev').rolling_mean(20).over('code').alias('_md')
        )
        .with_columns(
            (safe_div(pl.col('_tp') - pl.col('_sma_tp'), 0.015 * pl.col('_md')) / 100.0).alias("tc006_048")
        )
        .select(['trade_time', 'code', "tc006_048"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_048；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
