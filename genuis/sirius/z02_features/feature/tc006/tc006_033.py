"""
因子代号: tc006_033
原历史名: tc006_033
因子定义: 距 20 周期最低点的周期数 TS_ARGMIN(low, 20)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算距 20 周期最低点的周期数 BARS_FROM_LOW。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_033 列
    """
    shifts = [pl.col('low').shift(i).over('code').alias(f'_l_{i}') for i in range(20)]

    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(shifts)
        .with_columns(
            pl.col('low').rolling_min(20).over('code').alias('_min_l')
        )
        .with_columns(
            pl.min_horizontal([
                pl.when(pl.col(f'_l_{i}') == pl.col('_min_l')).then(float(i)).otherwise(20.0)
                for i in range(20)
            ]).alias("tc006_033")
        )
        .select(['trade_time', 'code', "tc006_033"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_033；输入含 trade_time、code、low。"""
    return calculate(df_lazy)
