"""
因子代号: tc001_015
原历史名: tc001_015
因子定义: 估波指标 Coppock = WMA(ROC(close, 14) + ROC(close, 11), 10)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算估波指标Coppock因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_015 列
    """

    # 计算步骤:
    # 1. 计算R(14) = (收盘价 - 14周期前收盘价) / 14周期前收盘价 * 100
    # 2. 计算R(11) = (收盘价 - 11周期前收盘价) / 11周期前收盘价 * 100
    # 3. RC = R(14) + R(11)
    # 4. COPPOCK = RC的10周期加权移动平均（用EMA近似）
    # 5. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 14周期前收盘价
            pl.col('close').shift(14).over('code').alias('close_14d'),
            # 11周期前收盘价
            pl.col('close').shift(11).over('code').alias('close_11d')
        )
        .with_columns(
            # R(14) = (收盘价 - 14周期前收盘价) / 14周期前收盘价 * 100
            pl.when(pl.col('close_14d') == 0)
            .then(0.0)
            .otherwise((pl.col('close') - pl.col('close_14d')) / pl.col('close_14d') * 100)
            .alias('r_14'),
            # R(11) = (收盘价 - 11周期前收盘价) / 11周期前收盘价 * 100
            pl.when(pl.col('close_11d') == 0)
            .then(0.0)
            .otherwise((pl.col('close') - pl.col('close_11d')) / pl.col('close_11d') * 100)
            .alias('r_11')
        )
        .with_columns(
            # RC = R(14) + R(11)
            (pl.col('r_14') + pl.col('r_11')).alias('rc')
        )
        .with_columns(
            # COPPOCK = RC的10周期加权移动平均（用EMA近似）
            pl.col('rc').ewm_mean(span=10).over('code').alias("tc001_015")
        )
        .select(['trade_time', 'code', "tc001_015"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
