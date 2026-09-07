"""
因子代号: tc006_019
原历史名: tc006_019
因子定义: 20周期已实现历史波动率 STDDEV(LOG_RETURN(close, 1), 20)
"""
import polars as pl

from feature.utils import log_return


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 20 周期已实现历史波动率 REALIZED_VOL。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_019 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            log_return('close').alias('_log_ret')
        )
        .with_columns(
            pl.col('_log_ret').rolling_std(20).over('code').alias("tc006_019")
        )
        .select(['trade_time', 'code', "tc006_019"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_019；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
