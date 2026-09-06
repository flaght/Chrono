"""
因子代号: tc006_051
原历史名: tc006_051
因子定义: 上涨 K 线成交量占 20 周期总成交量比例
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算上涨 K 线成交量占 20 周期总成交量比例 UP_VOL_RATIO。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_051 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(1).over('code').alias('_prev_close')
        )
        .with_columns(
            pl.when(pl.col('close') > pl.col('_prev_close'))
            .then(pl.col('volume'))
            .otherwise(0.0)
            .alias('_up_volume')
        )
        .with_columns(
            pl.col('_up_volume').rolling_sum(20).over('code').alias('_sum_up_vol'),
            pl.col('volume').rolling_sum(20).over('code').alias('_sum_total_vol'),
        )
        .with_columns(
            safe_div(pl.col('_sum_up_vol'), pl.col('_sum_total_vol')).alias("tc006_051")
        )
        .select(['trade_time', 'code', "tc006_051"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_051；输入含 trade_time、code、close、volume。"""
    return calculate(df_lazy)
