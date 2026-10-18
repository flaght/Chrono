"""
因子代号: tc006_040
原历史名: tc006_040
因子定义: 一目均衡表转换-基准差 (ConvLine - BaseLine) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算一目均衡表转换线与基准线差值比率 ICHIMOKU_SPAN。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_040 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('high').rolling_max(9).over('code').alias('_max_h9'),
            pl.col('low').rolling_min(9).over('code').alias('_min_l9'),
            pl.col('high').rolling_max(26).over('code').alias('_max_h26'),
            pl.col('low').rolling_min(26).over('code').alias('_min_l26'),
        )
        .with_columns(
            ((pl.col('_max_h9') + pl.col('_min_l9')) / 2.0).alias('_conv_line'),
            ((pl.col('_max_h26') + pl.col('_min_l26')) / 2.0).alias('_base_line'),
        )
        .with_columns(
            safe_div(pl.col('_conv_line') - pl.col('_base_line'), pl.col('close')).alias("tc006_040")
        )
        .select(['trade_time', 'code', "tc006_040"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_040；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
