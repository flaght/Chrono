"""
因子代号: tc001_027
原历史名: tc001_027
因子定义: 漂亮振荡器 PGO = (close - rolling_min(close, 10)) / (rolling_max(close, 10) - rolling_min(close, 10))
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算PGO因子（Pretty Good Oscillator变体）

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_027 列
    """

    # 计算步骤:
    # 1. 计算10周期最高价和最低价
    # 2. PGO = (收盘价 - 10周期最低价) / (10周期最高价 - 10周期最低价)
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 10周期最高价
            pl.col('high').rolling_max(10).over('code').alias('high_10d'),
            # 10周期最低价
            pl.col('low').rolling_min(10).over('code').alias('low_10d')
        )
        .with_columns(
            # PGO = (收盘价 - 10周期最低价) / (10周期最高价 - 10周期最低价)
            # 当最高价=最低价时，设为0.5
            pl.when(pl.col('high_10d') == pl.col('low_10d'))
            .then(0.5)
            .otherwise((pl.col('close') - pl.col('low_10d')) / 
                      (pl.col('high_10d') - pl.col('low_10d')))
            .alias("tc001_027")
        )
        .select(['trade_time', 'code', "tc001_027"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
