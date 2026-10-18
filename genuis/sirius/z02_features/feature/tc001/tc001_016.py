"""
因子代号: tc001_016
原历史名: tc001_016
因子定义: Elder牛熊力量指标 ERI = (high - EMA(close, 13)) + (low - EMA(close, 13))
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算Elder力指数(ERI)因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_016 列
    """

    # 计算步骤:
    # 1. 计算13周期EMA(收盘价)
    # 2. 牛力 = 最高价 - EMA
    # 3. 熊力 = 最低价 - EMA
    # 4. ERI = 牛力 + 熊力
    # 5. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 13周期指数移动平均线
            pl.col('close').ewm_mean(span=13).over('code').alias('ema_13')
        )
        .with_columns(
            # 牛力 = 最高价 - EMA
            (pl.col('high') - pl.col('ema_13')).alias('bull_power'),
            # 熊力 = 最低价 - EMA
            (pl.col('low') - pl.col('ema_13')).alias('bear_power')
        )
        .with_columns(
            # ERI = 牛力 + 熊力
            (pl.col('bull_power') + pl.col('bear_power')).alias("tc001_016")
        )
        .select(['trade_time', 'code', "tc001_016"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
