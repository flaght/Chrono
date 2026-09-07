"""
因子代号: tc006_049
原历史名: tc006_049
因子定义: 资金流量与价格 RSI 背离差 MFI(14) - RSI(14)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算资金流量与价格 RSI 背离差 MFI_RSI_DIFF。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_049 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0).alias('_tp'),
            pl.col('close').diff(1).over('code').alias('_diff_close'),
        )
        .with_columns(
            pl.col('_tp').shift(1).over('code').alias('_prev_tp'),
            (pl.col('_tp') * pl.col('volume')).alias('_raw_mf'),
            pl.when(pl.col('_diff_close') > 0).then(pl.col('_diff_close')).otherwise(0.0).alias('_gain'),
            pl.when(pl.col('_diff_close') < 0).then(-pl.col('_diff_close')).otherwise(0.0).alias('_loss'),
        )
        .with_columns(
            pl.when(pl.col('_tp') > pl.col('_prev_tp')).then(pl.col('_raw_mf')).otherwise(0.0).alias('_pos_mf'),
            pl.when(pl.col('_tp') < pl.col('_prev_tp')).then(pl.col('_raw_mf')).otherwise(0.0).alias('_neg_mf'),
            pl.col('_gain').ewm_mean(alpha=1.0/14.0, adjust=False).over('code').alias('_avg_gain'),
            pl.col('_loss').ewm_mean(alpha=1.0/14.0, adjust=False).over('code').alias('_avg_loss'),
        )
        .with_columns(
            pl.col('_pos_mf').rolling_sum(14).over('code').alias('_sum_pos'),
            pl.col('_neg_mf').rolling_sum(14).over('code').alias('_sum_neg'),
            safe_div(pl.col('_avg_gain'), pl.col('_avg_loss')).alias('_rs'),
        )
        .with_columns(
            safe_div(pl.col('_sum_pos'), pl.col('_sum_neg')).alias('_mfr'),
            pl.when(pl.col('_avg_loss') == 0).then(100.0).otherwise(100.0 - 100.0 / (1.0 + pl.col('_rs'))).alias('_rsi14'),
        )
        .with_columns(
            pl.when(pl.col('_sum_neg') == 0).then(100.0).otherwise(100.0 - 100.0 / (1.0 + pl.col('_mfr'))).alias('_mfi14')
        )
        .with_columns(
            (pl.col('_mfi14') - pl.col('_rsi14')).alias("tc006_049")
        )
        .select(['trade_time', 'code', "tc006_049"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_049；输入含 trade_time、code、high、low、close、volume。"""
    return calculate(df_lazy)
