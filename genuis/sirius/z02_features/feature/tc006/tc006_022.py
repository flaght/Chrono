"""
因子代号: tc006_022
原历史名: tc006_022
因子定义: 短期与长期标准差比值 STDDEV(close, 5) / STDDEV(close, 20)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算短期与长期标准差比值 VOL_EXPANSION。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_022 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').rolling_std(5).over('code').alias('_std5'),
            pl.col('close').rolling_std(20).over('code').alias('_std20'),
        )
        .with_columns(
            safe_div(pl.col('_std5'), pl.col('_std20')).alias("tc006_022")
        )
        .select(['trade_time', 'code', "tc006_022"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_022；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
