"""
因子代号: tc006_011
原历史名: tc006_011
因子定义: 14周期威廉指标 WILLR(high, low, close, 14)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 14 周期威廉指标 WILLR。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_011 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('high').rolling_max(14).over('code').alias('_max_h'),
            pl.col('low').rolling_min(14).over('code').alias('_min_l'),
        )
        .with_columns(
            (-100.0 * safe_div(pl.col('_max_h') - pl.col('close'), pl.col('_max_h') - pl.col('_min_l'))).alias("tc006_011")
        )
        .select(['trade_time', 'code', "tc006_011"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_011；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
