"""
因子代号: tc006_047
原历史名: tc006_047
因子定义: 14周期 RSI 远离中性区极端度 ABS(RSI(close, 14) - 50)
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 14 周期 RSI 远离中性区极端度 RSI_ZONE。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_047 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').diff(1).over('code').alias('_diff')
        )
        .with_columns(
            pl.when(pl.col('_diff') > 0).then(pl.col('_diff')).otherwise(0.0).alias('_gain'),
            pl.when(pl.col('_diff') < 0).then(-pl.col('_diff')).otherwise(0.0).alias('_loss'),
        )
        .with_columns(
            pl.col('_gain').ewm_mean(alpha=1.0/14.0, adjust=False).over('code').alias('_avg_gain'),
            pl.col('_loss').ewm_mean(alpha=1.0/14.0, adjust=False).over('code').alias('_avg_loss'),
        )
        .with_columns(
            safe_div(pl.col('_avg_gain'), pl.col('_avg_loss')).alias('_rs')
        )
        .with_columns(
            pl.when(pl.col('_avg_loss') == 0)
            .then(100.0)
            .otherwise(100.0 - 100.0 / (1.0 + pl.col('_rs')))
            .alias('_rsi14')
        )
        .with_columns(
            (pl.col('_rsi14') - 50.0).abs().alias("tc006_047")
        )
        .select(['trade_time', 'code', "tc006_047"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_047；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
