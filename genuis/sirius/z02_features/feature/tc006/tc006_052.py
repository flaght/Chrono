"""
因子代号: tc006_052
原历史名: tc006_052
因子定义: 10周期考夫曼价格效率比 ABS(close - DELAY(close, 10)) / SUM(ABS(close - DELAY(close, 1)), 10)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 10 周期考夫曼价格效率比 EFFICIENCY_RATIO。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_052 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            (pl.col('close') - pl.col('close').shift(10).over('code')).abs().alias('_net_change'),
            (pl.col('close') - pl.col('close').shift(1).over('code')).abs().alias('_step_change'),
        )
        .with_columns(
            pl.col('_step_change').rolling_sum(10).over('code').alias('_sum_change')
        )
        .with_columns(
            safe_div(pl.col('_net_change'), pl.col('_sum_change')).alias("tc006_052")
        )
        .select(['trade_time', 'code', "tc006_052"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_052；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
