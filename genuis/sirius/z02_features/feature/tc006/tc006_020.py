"""
因子代号: tc006_020
原历史名: tc006_020
因子定义: 单根 K 线真实波幅占价格比例 TRUE_RANGE(high, low, close) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算单根 K 线真实波幅占价格比例 TR_PCT。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_020 列
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
            safe_div(pl.col('_tr'), pl.col('close')).alias("tc006_020")
        )
        .select(['trade_time', 'code', "tc006_020"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_020；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
