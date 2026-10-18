"""
因子代号: tc006_012
原历史名: tc006_012
因子定义: 15周期三重指数平滑变化率 TRIX(close, 15)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 15 周期三重指数平滑变化率 TRIX。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_012 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').ewm_mean(span=15, adjust=False).over('code').alias('_ema1')
        )
        .with_columns(
            pl.col('_ema1').ewm_mean(span=15, adjust=False).over('code').alias('_ema2')
        )
        .with_columns(
            pl.col('_ema2').ewm_mean(span=15, adjust=False).over('code').alias('_ema3')
        )
        .with_columns(
            pl.col('_ema3').shift(1).over('code').alias('_prev_ema3')
        )
        .with_columns(
            (100.0 * safe_div(pl.col('_ema3') - pl.col('_prev_ema3'), pl.col('_prev_ema3'))).alias("tc006_012")
        )
        .select(['trade_time', 'code', "tc006_012"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_012；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
