"""
因子代号: tc006_023
原历史名: tc006_023
因子定义: 布林带挤压度 4 * STDDEV(close, 20) / SMA(close, 20)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算布林带挤压度 BOLL_SQUEEZE。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_023 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').rolling_std(20).over('code').alias('_std20'),
            pl.col('close').rolling_mean(20).over('code').alias('_sma20'),
        )
        .with_columns(
            safe_div(4.0 * pl.col('_std20'), pl.col('_sma20')).alias("tc006_023")
        )
        .select(['trade_time', 'code', "tc006_023"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_023；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
