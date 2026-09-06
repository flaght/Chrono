"""
因子代号: tc001_014
原历史名: tc001_014
因子定义: 佳庆指标 CHO = EMA(AD, 10) - EMA(AD, 3)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算佳庆指标CHO因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_014 列
    """

    # 计算步骤:
    # 1. 计算每周期AD增量 = 成交量 * (2*收盘价 - 最高价 - 最低价) / (最高价 + 最低价)
    # 2. 计算AD的10周期EMA和3周期EMA
    # 3. CHO = EMA(AD, 10) - EMA(AD, 3)
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 每周期AD增量 = 成交量 * (2*收盘价 - 最高价 - 最低价) / (最高价 + 最低价)
            # 当最高价+最低价=0时，AD设为0
            pl.when(pl.col('high') + pl.col('low') == 0)
            .then(0.0)
            .otherwise(pl.col('volume') * 
                      (2 * pl.col('close') - pl.col('high') - pl.col('low')) / 
                      (pl.col('high') + pl.col('low')))
            .alias('daily_ad')
        )
        .with_columns(
            # AD的10周期EMA
            pl.col('daily_ad').ewm_mean(span=10).over('code').alias('ema_ad_10'),
            # AD的3周期EMA
            pl.col('daily_ad').ewm_mean(span=3).over('code').alias('ema_ad_3')
        )
        .with_columns(
            # CHO = EMA(AD, 10) - EMA(AD, 3)
            (pl.col('ema_ad_10') - pl.col('ema_ad_3')).alias("tc001_014")
        )
        .select(['trade_time', 'code', "tc001_014"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
