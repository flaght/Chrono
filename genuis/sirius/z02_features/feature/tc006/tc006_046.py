"""
因子代号: tc006_046
原历史名: tc006_046
因子定义: 20周期三重指数移动平均偏离度 (close - TEMA(close, 20)) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 20 周期三重指数移动平均偏离度 TEMA20_DEV。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_046 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').ewm_mean(span=20, adjust=False).over('code').alias('_ema1')
        )
        .with_columns(
            pl.col('_ema1').ewm_mean(span=20, adjust=False).over('code').alias('_ema2')
        )
        .with_columns(
            pl.col('_ema2').ewm_mean(span=20, adjust=False).over('code').alias('_ema3')
        )
        .with_columns(
            (3.0 * pl.col('_ema1') - 3.0 * pl.col('_ema2') + pl.col('_ema3')).alias('_tema')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_tema'), pl.col('close')).alias("tc006_046")
        )
        .select(['trade_time', 'code', "tc006_046"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_046；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
