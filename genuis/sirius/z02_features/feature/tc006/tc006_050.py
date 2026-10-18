"""
因子代号: tc006_050
原历史名: tc006_050
因子定义: 5周期量价背离指标 SIGN(ROC(close, 5)) * (-1) * ROC(volume, 5) / 100
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 5 周期量价背离指标 VOL_PRICE_DIV。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_050 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(5).over('code').alias('_c5'),
            pl.col('volume').shift(5).over('code').alias('_v5'),
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('_c5'), pl.col('_c5')).alias('_roc_c5'),
            safe_div(pl.col('volume') - pl.col('_v5'), pl.col('_v5')).alias('_roc_v5'),
        )
        .with_columns(
            (pl.col('_roc_c5').sign() * (-1.0) * pl.col('_roc_v5') / 100.0).alias("tc006_050")
        )
        .select(['trade_time', 'code', "tc006_050"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_050；输入含 trade_time、code、close、volume。"""
    return calculate(df_lazy)
