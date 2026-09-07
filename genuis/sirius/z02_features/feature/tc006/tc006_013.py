"""
因子代号: tc006_013
原历史名: tc006_013
因子定义: 归一化 10 周期动量 (close - close.shift(10)) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算归一化 10 周期动量占价格比例 MOM10_NORM。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_013 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(10).over('code').alias('_prev_close')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_prev_close'), pl.col('close')).alias("tc006_013")
        )
        .select(['trade_time', 'code', "tc006_013"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_013；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
