"""
因子代号: tc006_015
原历史名: tc006_015
因子定义: 动量加速度 ROC(close, 3) - ROC(close, 10)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算动量加速度 MOM_ACCEL。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_015 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(3).over('code').alias('_c3'),
            pl.col('close').shift(10).over('code').alias('_c10'),
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_c3'), pl.col('_c3')).alias('_roc3'),
            safe_div(pl.col('close') - pl.col('_c10'), pl.col('_c10')).alias('_roc10'),
        )
        .with_columns(
            (pl.col('_roc3') - pl.col('_roc10')).alias("tc006_015")
        )
        .select(['trade_time', 'code', "tc006_015"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_015；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
