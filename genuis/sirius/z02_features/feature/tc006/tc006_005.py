"""
因子代号: tc006_005
原历史名: tc006_005
因子定义: 考夫曼自适应均线偏离度 (close - KAMA(close, 20)) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 KAMA20 偏离度因子。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_005 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            (pl.col('close') - pl.col('close').shift(20).over('code')).abs().alias('_change'),
            (pl.col('close') - pl.col('close').shift(1).over('code')).abs().alias('_diff1'),
        )
        .with_columns(
            pl.col('_diff1').rolling_sum(20).over('code').alias('_volatility')
        )
        .with_columns(
            safe_div(pl.col('_change'), pl.col('_volatility')).alias('_er')
        )
        .with_columns(
            ((pl.col('_er') * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0) ** 2).alias('_sc')
        )
        .with_columns(
            # 自适应平滑权重加权移动平均近似
            (pl.col('close') * pl.col('_sc')).rolling_sum(20).over('code').alias('_kama_num'),
            pl.col('_sc').rolling_sum(20).over('code').alias('_kama_den'),
        )
        .with_columns(
            safe_div(pl.col('_kama_num'), pl.col('_kama_den')).alias('_kama')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_kama'), pl.col('close')).alias("tc006_005")
        )
        .select(['trade_time', 'code', "tc006_005"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_005；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
