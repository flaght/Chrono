"""
因子代号: tc001_009
原历史名: tc001_009
因子定义: 6周期乖离率 BIAS_6 = (close - SMA(close, 6)) / SMA(close, 6) * 100
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算6周期乖离率因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_009 列
    """

    # 计算步骤:
    # 1. 计算6周期移动平均线
    # 2. 乖离率 = (收盘价 - 均线) / 均线 * 100
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 6周期移动平均线
            pl.col('close').rolling_mean(6).over('code').alias('ma_6')
        )
        .with_columns(
            # 乖离率 = (收盘价 - 均线) / 均线 * 100
            ((pl.col('close') - pl.col('ma_6')) / pl.col('ma_6') * 100)
            .alias("tc001_009")
        )
        .select(['trade_time', 'code', "tc001_009"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
