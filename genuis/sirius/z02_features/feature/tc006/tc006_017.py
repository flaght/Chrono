"""
因子代号: tc006_017
原历史名: tc006_017
因子定义: 归一化平均真实波幅百分比 ATR(high, low, close, 14) / close
"""
import polars as pl

from feature.utils import safe_div


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 14 周期归一化 ATR 百分比 NATR14。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_017 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').shift(1).over('code').alias('_prev_close')
        )
        .with_columns(
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('_prev_close')).abs(),
                (pl.col('low') - pl.col('_prev_close')).abs(),
            ]).alias('_tr')
        )
        .with_columns(
            pl.col('_tr').rolling_mean(14).over('code').alias('_atr14')
        )
        .with_columns(
            (100.0 * safe_div(pl.col('_atr14'), pl.col('close'))).alias("tc006_017")
        )
        .select(['trade_time', 'code', "tc006_017"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_017；输入含 trade_time、code、high、low、close。"""
    return calculate(df_lazy)
