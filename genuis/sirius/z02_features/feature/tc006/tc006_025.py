"""
因子代号: tc006_025
原历史名: tc006_025
因子定义: 20周期蔡金资金流量指标 CMF(high, low, close, volume, 20)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 20 周期蔡金资金流量指标 CMF。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_025 列
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
                ) * pl.col('volume')
            )
            .alias('_mfv')
        )
        .with_columns(
            pl.col('_mfv').rolling_sum(20).over('code').alias('_sum_mfv'),
            pl.col('volume').rolling_sum(20).over('code').alias('_sum_vol'),
        )
        .with_columns(
            safe_div(pl.col('_sum_mfv'), pl.col('_sum_vol')).alias("tc006_025")
        )
        .select(['trade_time', 'code', "tc006_025"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_025；输入含 trade_time、code、high、low、close、volume。"""
    return calculate(df_lazy)
