"""
因子代号: tc006_043
原历史名: tc006_043
因子定义: 200周期 EMA 偏离度 (close - EMA(close, 200)) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 200 周期 EMA 偏离度 EMA200_DEV。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_043 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').ewm_mean(span=200, adjust=False).over('code').alias('_ema200')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_ema200'), pl.col('close')).alias("tc006_043")
        )
        .select(['trade_time', 'code', "tc006_043"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_043；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
