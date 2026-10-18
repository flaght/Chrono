"""
因子代号: tc006_039
原历史名: tc006_039
因子定义: 一目均衡表基准线偏离度 (close - (MAX(high, 26) + MIN(low, 26)) / 2) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算一目均衡表基准线偏离度 ICHIMOKU_BASE_DEV。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_039 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('high').rolling_max(26).over('code').alias('_max_h26'),
            pl.col('low').rolling_min(26).over('code').alias('_min_l26'),
        )
        .with_columns(
            ((pl.col('_max_h26') + pl.col('_min_l26')) / 2.0).alias('_base_line')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_base_line'), pl.col('close')).alias("tc006_039")
        )
        .select(['trade_time', 'code', "tc006_039"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_039；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
