"""
因子代号: tc006_027
原历史名: tc006_027
因子定义: 累积/派发线 CUM_SUM(MFM * volume)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算全历史累积/派发线 AD_LINE。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_027 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.when(pl.col('high') == pl.col('low'))
            .then(0.0)
            .otherwise(
                safe_div(
                    (pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close')),
                    pl.col('high') - pl.col('low')
                )
            )
            .alias('_mfm')
        )
        .with_columns(
            (pl.col('_mfm') * pl.col('volume')).alias('_ad_flow')
        )
        .with_columns(
            pl.col('_ad_flow').cum_sum().over('code').alias("tc006_027")
        )
        .select(['trade_time', 'code', "tc006_027"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_027；输入含 trade_time、code、high、low、close、volume。"""
    return calculate(df_lazy)
