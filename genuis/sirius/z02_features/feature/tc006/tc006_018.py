"""
因子代号: tc006_018
原历史名: tc006_018
因子定义: 波动率扩张收缩比 ATR(5) / ATR(20)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算波动率扩张/收缩比 ATR_RATIO。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_018 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(1).over('code').alias('_prev_close')
        )
        .with_columns(
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('_prev_close')).abs(),
                (pl.col('low') - pl.col('_prev_close')).abs(),
            ]).alias('_tr')
        )
        .with_columns(
            pl.col('_tr').rolling_mean(5).over('code').alias('_atr5'),
            pl.col('_tr').rolling_mean(20).over('code').alias('_atr20'),
        )
        .with_columns(
            safe_div(pl.col('_atr5'), pl.col('_atr20')).alias("tc006_018")
        )
        .select(['trade_time', 'code', "tc006_018"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_018；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
