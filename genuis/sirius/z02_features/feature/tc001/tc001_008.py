"""
因子代号: tc001_008
原历史名: tc001_008
因子定义: 26周期乖离率 BIAS_26 = (close - SMA(close, 26)) / SMA(close, 26) * 100
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算26周期乖离率因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_008 列
    """

    # 计算步骤:
    # 1. 计算26周期简单移动平均线
    # 2. BIAS = (收盘价 - 均线) / 均线 * 100
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 26周期简单移动平均线
            pl.col('close').rolling_mean(26).over('code').alias('sma_26')
        )
        .with_columns(
            # BIAS = (收盘价 - 均线) / 均线 * 100
            # 当均线=0时，设为0
            pl.when(pl.col('sma_26') == 0)
            .then(0.0)
            .otherwise((pl.col('close') - pl.col('sma_26')) / pl.col('sma_26') * 100)
            .alias("tc001_008")
        )
        .select(['trade_time', 'code', "tc001_008"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
