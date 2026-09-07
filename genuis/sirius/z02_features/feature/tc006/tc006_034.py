"""
因子代号: tc006_034
原历史名: tc006_034
因子定义: 10周期线性衰减加权动量 DECAYLINEAR(ROC(close, 1), 10)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 10 周期线性衰减加权动量 DECAY_MOM。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_034 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(1).over('code').alias('_c1')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_c1'), pl.col('_c1')).alias('_roc1')
        )
        .with_columns(
            [pl.col('_roc1').shift(i).over('code').alias(f'_r_{i}') for i in range(10)]
        )
        .with_columns(
            (sum((10 - i) * pl.col(f'_r_{i}') for i in range(10)) / 55.0).alias("tc006_034")
        )
        .select(['trade_time', 'code', "tc006_034"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_034；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
