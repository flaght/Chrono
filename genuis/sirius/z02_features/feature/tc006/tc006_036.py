"""
因子代号: tc006_036
原历史名: tc006_036
因子定义: 成交量滚动 20 周期 Z-Score (volume - SMA(volume, 20)) / STDDEV(volume, 20)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算成交量滚动 20 周期 Z-Score NORM_VOLUME。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_036 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('volume').rolling_mean(20).over('code').alias('_sma20'),
            pl.col('volume').rolling_std(20).over('code').alias('_std20'),
        )
        .with_columns(
            safe_div(pl.col('volume') - pl.col('_sma20'), pl.col('_std20')).alias("tc006_036")
        )
        .select(['trade_time', 'code', "tc006_036"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_036；输入含 trade_time、code、volume。"""
    return calculate(df_lazy)
