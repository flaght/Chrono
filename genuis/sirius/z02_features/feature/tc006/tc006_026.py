"""
因子代号: tc006_026
原历史名: tc006_026
因子定义: 14周期资金流量指标 MFI(high, low, close, volume, 14)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 14 周期资金流量指标 MFI。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_026 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0).alias('_tp')
        )
        .with_columns(
            pl.col('_tp').shift(1).over('code').alias('_prev_tp'),
            (pl.col('_tp') * pl.col('volume')).alias('_raw_mf'),
        )
        .with_columns(
            pl.when(pl.col('_tp') > pl.col('_prev_tp'))
            .then(pl.col('_raw_mf'))
            .otherwise(0.0)
            .alias('_pos_mf'),
            pl.when(pl.col('_tp') < pl.col('_prev_tp'))
            .then(pl.col('_raw_mf'))
            .otherwise(0.0)
            .alias('_neg_mf'),
        )
        .with_columns(
            pl.col('_pos_mf').rolling_sum(14).over('code').alias('_sum_pos'),
            pl.col('_neg_mf').rolling_sum(14).over('code').alias('_sum_neg'),
        )
        .with_columns(
            safe_div(pl.col('_sum_pos'), pl.col('_sum_neg')).alias('_mfr')
        )
        .with_columns(
            pl.when(pl.col('_sum_neg') == 0)
            .then(100.0)
            .otherwise(100.0 - 100.0 / (1.0 + pl.col('_mfr')))
            .alias("tc006_026")
        )
        .select(['trade_time', 'code', "tc006_026"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_026；输入含 trade_time、code、high、low、close、volume。"""
    return calculate(df_lazy)
