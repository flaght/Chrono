"""
因子代号: tc006_042
原历史名: tc006_042
因子定义: 相对经典枢轴点距离 (close - (high + low + close) / 3) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算相对经典枢轴点距离 PIVOT_DIST。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_042 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0).alias('_pivot')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_pivot'), pl.col('close')).alias("tc006_042")
        )
        .select(['trade_time', 'code', "tc006_042"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_042；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
