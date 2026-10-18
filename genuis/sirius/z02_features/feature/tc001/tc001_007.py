"""
因子代号: tc001_007
原历史名: tc001_007
因子定义: 多空指数 BBI = (SMA(close, 3) + SMA(close, 6) + SMA(close, 12) + SMA(close, 24)) / 4
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算多空指数BBI因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_007 列
    """

    # 计算步骤:
    # 1. 计算3周期、6周期、12周期、20周期移动平均线
    # 2. BBI = (MA3 + MA6 + MA12 + MA20) / 4
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 3周期移动平均线
            pl.col('close').rolling_mean(3).over('code').alias('ma_3'),
            # 6周期移动平均线
            pl.col('close').rolling_mean(6).over('code').alias('ma_6'),
            # 12周期移动平均线
            pl.col('close').rolling_mean(12).over('code').alias('ma_12'),
            # 20周期移动平均线
            pl.col('close').rolling_mean(20).over('code').alias('ma_20')
        )
        .with_columns(
            # 多空指数 BBI = (MA3 + MA6 + MA12 + MA20) / 4
            ((pl.col('ma_3') + pl.col('ma_6') + pl.col('ma_12') + pl.col('ma_20')) / 4)
            .alias("tc001_007")
        )
        .select(['trade_time', 'code', "tc001_007"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
