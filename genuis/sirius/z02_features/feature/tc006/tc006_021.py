"""
因子代号: tc006_021
原历史名: tc006_021
因子定义: 收盘价在单根 K 线振幅中的相对位置 (close - low) / (high - low)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算收盘价在单根 K 线振幅中的相对位置 CLOSE_RANGE。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_021 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.when(pl.col('high') == pl.col('low'))
            .then(0.5)
            .otherwise(safe_div(pl.col('close') - pl.col('low'), pl.col('high') - pl.col('low')))
            .alias("tc006_021")
        )
        .select(['trade_time', 'code', "tc006_021"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_021；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
