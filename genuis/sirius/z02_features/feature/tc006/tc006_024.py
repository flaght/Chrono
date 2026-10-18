"""
因子代号: tc006_024
原历史名: tc006_024
因子定义: 单根 K 线日内高低振幅比例 (high - low) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算单根 K 线日内高低振幅比例 HIGH_LOW_PCT。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_024 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            safe_div(pl.col('high') - pl.col('low'), pl.col('close')).alias("tc006_024")
        )
        .select(['trade_time', 'code', "tc006_024"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_024；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
