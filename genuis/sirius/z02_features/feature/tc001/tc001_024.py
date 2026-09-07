"""
因子代号: tc001_024
原历史名: tc001_024
因子定义: 梅斯线 Mass Index = rolling_sum(EMA(high - low, 9) / EMA(EMA(high - low, 9), 9), 25)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算梅斯线Mass Index因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_024 列
    """

    # 计算步骤:
    # 1. 计算振幅 = 最高价 - 最低价
    # 2. 计算振幅的9周期EMA
    # 3. 计算振幅EMA的9周期EMA
    # 4. 计算比值 = EMA1 / EMA2
    # 5. 计算25周期比值的累加和作为Mass Index
    # 6. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 振幅 = 最高价 - 最低价
            (pl.col('high') - pl.col('low')).alias('amplitude')
        )
        .with_columns(
            # 振幅的9周期EMA
            pl.col('amplitude').ewm_mean(span=9).over('code').alias('ema1'),
        )
        .with_columns(
            # 振幅EMA的9周期EMA
            pl.col('ema1').ewm_mean(span=9).over('code').alias('ema2')
        )
        .with_columns(
            # 比值 = EMA1 / EMA2
            # 当EMA2=0时，设为1
            pl.when(pl.col('ema2') == 0)
            .then(1.0)
            .otherwise(pl.col('ema1') / pl.col('ema2'))
            .alias('ratio')
        )
        .with_columns(
            # 25周期比值累加和 = Mass Index
            pl.col('ratio').rolling_sum(25).over('code').alias("tc001_024")
        )
        .select(['trade_time', 'code', "tc001_024"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
