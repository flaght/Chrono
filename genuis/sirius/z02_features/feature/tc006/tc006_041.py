"""
因子代号: tc006_041
原历史名: tc006_041
因子定义: 肯特纳通道相对位置 (close - EMA(close, 20)) / (2 * ATR(14))
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算肯特纳通道相对位置 KELTNER_POS。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_041 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(1).over('code').alias('_prev_close'),
            pl.col('close').ewm_mean(span=20, adjust=False).over('code').alias('_ema20'),
        )
        .with_columns(
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('_prev_close')).abs(),
                (pl.col('low') - pl.col('_prev_close')).abs(),
            ]).alias('_tr')
        )
        .with_columns(
            pl.col('_tr').rolling_mean(14).over('code').alias('_atr14')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_ema20'), 2.0 * pl.col('_atr14')).alias("tc006_041")
        )
        .select(['trade_time', 'code', "tc006_041"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_041；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
