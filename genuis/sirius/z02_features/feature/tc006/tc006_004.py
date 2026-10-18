"""
因子代号: tc006_004
原历史名: tc006_004
因子定义: Hull 移动平均偏离度 (close - HMA(close, 20)) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 HMA20 偏离度因子。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_004 列
    """
    wma10_expr = sum((10 - i) * pl.col('close').shift(i).over('code') for i in range(10)) / 55.0
    wma20_expr = sum((20 - i) * pl.col('close').shift(i).over('code') for i in range(20)) / 210.0

    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            wma10_expr.alias('_wma10'),
            wma20_expr.alias('_wma20'),
        )
        .with_columns(
            (2.0 * pl.col('_wma10') - pl.col('_wma20')).alias('_raw_hma')
        )
        .with_columns(
            (sum((4 - i) * pl.col('_raw_hma').shift(i).over('code') for i in range(4)) / 10.0).alias('_hma')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_hma'), pl.col('close')).alias("tc006_004")
        )
        .select(['trade_time', 'code', "tc006_004"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_004；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
