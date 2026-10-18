"""
因子代号: tc006_054
原历史名: tc006_054
因子定义: K 线影线与实体波动比 (high - low) / ABS(close - open)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 K 线影线与实体波动比 INTRABAR_VOL。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、open、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_054 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            (pl.col('close') - pl.col('open')).abs().alias('_body')
        )
        .with_columns(
            safe_div(pl.col('high') - pl.col('low'), pl.col('_body')).alias("tc006_054")
        )
        .select(['trade_time', 'code', "tc006_054"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_054；输入含 trade_time、code、open、high、low、close。"""
    return calculate(df_lazy)
