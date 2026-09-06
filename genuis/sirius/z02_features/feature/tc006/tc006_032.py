"""
因子代号: tc006_032
原历史名: tc006_032
因子定义: 距 20 周期最高点的周期数 TS_ARGMAX(high, 20)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算距 20 周期最高点的周期数 BARS_FROM_HIGH。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_032 列
    """
    shifts = [pl.col('high').shift(i).over('code').alias(f'_h_{i}') for i in range(20)]

    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(shifts)
        .with_columns(
            pl.col('high').rolling_max(20).over('code').alias('_max_h')
        )
        .with_columns(
            pl.min_horizontal([
                pl.when(pl.col(f'_h_{i}') == pl.col('_max_h')).then(float(i)).otherwise(20.0)
                for i in range(20)
            ]).alias("tc006_032")
        )
        .select(['trade_time', 'code', "tc006_032"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_032；输入含 trade_time、code、high。"""
    return calculate(df_lazy)
