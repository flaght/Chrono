"""
因子代号: tc006_030
原历史名: tc006_030
因子定义: 20周期对数收益率偏度 TS_SKEW(close, 20)
"""
import polars as pl

from feature.utils import log_return, safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 20 周期收益率偏度 RETURN_SKEW。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_030 列
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
            ((pl.col('_ret') - pl.col('_mean')) ** 3).alias('_diff3')
        )
        .with_columns(
            pl.col('_diff3').rolling_mean(20).over('code').alias('_m3')
        )
        .with_columns(
            safe_div(pl.col('_m3'), pl.col('_std') ** 3).alias("tc006_030")
        )
        .select(['trade_time', 'code', "tc006_030"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_030；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
