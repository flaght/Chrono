"""
因子代号: tc006_031
原历史名: tc006_031
因子定义: 20周期对数收益率峰度 TS_KURT(close, 20)
"""
import polars as pl

from feature.utils import log_return, safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 20 周期收益率超额峰度 RETURN_KURT。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_031 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            log_return('close').alias('_ret')
        )
        .with_columns(
            pl.col('_ret').rolling_mean(20).over('code').alias('_mean'),
            pl.col('_ret').rolling_std(20).over('code').alias('_std'),
        )
        .with_columns(
            ((pl.col('_ret') - pl.col('_mean')) ** 4).alias('_diff4')
        )
        .with_columns(
            pl.col('_diff4').rolling_mean(20).over('code').alias('_m4')
        )
        .with_columns(
            (safe_div(pl.col('_m4'), pl.col('_std') ** 4) - 3.0).alias("tc006_031")
        )
        .select(['trade_time', 'code', "tc006_031"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_031；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
