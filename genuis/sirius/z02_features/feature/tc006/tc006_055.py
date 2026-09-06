"""
因子代号: tc006_055
原历史名: tc006_055
因子定义: K 线实体占振幅比例 ABS(close - open) / (high - low)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 K 线实体占振幅比例 BODY_RATIO。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、open、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_055 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.when(pl.col('high') == pl.col('low'))
            .then(0.0)
            .otherwise(
                safe_div((pl.col('close') - pl.col('open')).abs(), pl.col('high') - pl.col('low'))
            )
            .alias("tc006_055")
        )
        .select(['trade_time', 'code', "tc006_055"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_055；输入含 trade_time、code、open、high、low、close。"""
    return calculate(df_lazy)
