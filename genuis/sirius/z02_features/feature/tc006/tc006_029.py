"""
因子代号: tc006_029
原历史名: tc006_029
因子定义: 价格滚动 20 周期 Z-Score (close - SMA(close, 20)) / STDDEV(close, 20)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算价格滚动 20 周期 Z-Score NORM_PRICE。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_029 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').rolling_mean(20).over('code').alias('_sma20'),
            pl.col('close').rolling_std(20).over('code').alias('_std20'),
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_sma20'), pl.col('_std20')).alias("tc006_029")
        )
        .select(['trade_time', 'code', "tc006_029"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_029；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
