"""
因子代号: tc006_007
原历史名: tc006_007
因子定义: 正负方向指标差值 PLUS_DI(14) - MINUS_DI(14)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 14 周期正负方向指标差值 DI_SPREAD。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_007 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(1).over('code').alias('_prev_close'),
            pl.col('high').shift(1).over('code').alias('_prev_high'),
            pl.col('low').shift(1).over('code').alias('_prev_low'),
        )
        .with_columns(
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('_prev_close')).abs(),
                (pl.col('low') - pl.col('_prev_close')).abs(),
            ]).alias('_tr'),
            (pl.col('high') - pl.col('_prev_high')).alias('_up_move'),
            (pl.col('_prev_low') - pl.col('low')).alias('_down_move'),
        )
        .with_columns(
            pl.when((pl.col('_up_move') > pl.col('_down_move')) & (pl.col('_up_move') > 0))
            .then(pl.col('_up_move'))
            .otherwise(0.0)
            .alias('_plus_dm'),
            pl.when((pl.col('_down_move') > pl.col('_up_move')) & (pl.col('_down_move') > 0))
            .then(pl.col('_down_move'))
            .otherwise(0.0)
            .alias('_minus_dm'),
        )
        .with_columns(
            pl.col('_tr').ewm_mean(alpha=1.0/14.0, adjust=False).over('code').alias('_tr_smooth'),
            pl.col('_plus_dm').ewm_mean(alpha=1.0/14.0, adjust=False).over('code').alias('_plus_dm_smooth'),
            pl.col('_minus_dm').ewm_mean(alpha=1.0/14.0, adjust=False).over('code').alias('_minus_dm_smooth'),
        )
        .with_columns(
            (100.0 * safe_div(pl.col('_plus_dm_smooth'), pl.col('_tr_smooth'))).alias('_plus_di'),
            (100.0 * safe_div(pl.col('_minus_dm_smooth'), pl.col('_tr_smooth'))).alias('_minus_di'),
        )
        .with_columns(
            (pl.col('_plus_di') - pl.col('_minus_di')).alias("tc006_007")
        )
        .select(['trade_time', 'code', "tc006_007"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_007；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
