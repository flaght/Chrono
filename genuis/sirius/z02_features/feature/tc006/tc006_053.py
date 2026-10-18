"""
因子代号: tc006_053
原历史名: tc006_053
因子定义: Amihud 非流动性冲击因子 ABS(ROC(close, 1)) / volume
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 Amihud 非流动性冲击因子 AMIHUD_ILLIQ。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_053 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(1).over('code').alias('_prev_close')
        )
        .with_columns(
            safe_div((pl.col('close') - pl.col('_prev_close')).abs(), pl.col('_prev_close')).alias('_abs_roc1')
        )
        .with_columns(
            safe_div(pl.col('_abs_roc1'), pl.col('volume')).alias("tc006_053")
        )
        .select(['trade_time', 'code', "tc006_053"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_053；输入含 trade_time、code、close、volume。"""
    return calculate(df_lazy)
