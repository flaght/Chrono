"""
因子代号: tc006_028
原历史名: tc006_028
因子定义: 成交量均线趋势 EMA(volume, 7) / EMA(volume, 21) - 1
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算成交量均线趋势 VOL_TREND。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_028 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('volume').ewm_mean(span=7, adjust=False).over('code').alias('_ema7'),
            pl.col('volume').ewm_mean(span=21, adjust=False).over('code').alias('_ema21'),
        )
        .with_columns(
            (safe_div(pl.col('_ema7'), pl.col('_ema21')) - 1.0).alias("tc006_028")
        )
        .select(['trade_time', 'code', "tc006_028"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_028；输入含 trade_time、code、volume。"""
    return calculate(df_lazy)
