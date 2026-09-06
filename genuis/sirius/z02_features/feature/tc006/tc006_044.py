"""
因子代号: tc006_044
原历史名: tc006_044
因子定义: 50周期 SMA 偏离度 (close - SMA(close, 50)) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 50 周期 SMA 偏离度 SMA50_DEV。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_044 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').rolling_mean(50).over('code').alias('_sma50')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_sma50'), pl.col('close')).alias("tc006_044")
        )
        .select(['trade_time', 'code', "tc006_044"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_044；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
