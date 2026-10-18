"""
因子代号: tc001_006
原历史名: tc001_006
因子定义: 动量震荡指标 AO = SMA((high + low) / 2, 5) - SMA((high + low) / 2, 34)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算绝佳振荡器(Awesome Oscillator)因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_006 列
    """

    # 计算步骤:
    # 1. 计算中间价 hl2 = (最高价 + 最低价) / 2
    # 2. 计算 hl2 的5周期简单移动平均
    # 3. 计算 hl2 的34周期简单移动平均
    # 4. AO = 5周期均线 - 34周期均线
    # 5. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 中间价 = (最高价 + 最低价) / 2
            ((pl.col('high') + pl.col('low')) / 2).alias('hl2')
        )
        .with_columns(
            # 5周期简单移动平均
            pl.col('hl2').rolling_mean(5).over('code').alias('sma_5'),
            # 34周期简单移动平均
            pl.col('hl2').rolling_mean(34).over('code').alias('sma_34')
        )
        .with_columns(
            # AO = 5周期均线 - 34周期均线
            (pl.col('sma_5') - pl.col('sma_34')).alias("tc001_006")
        )
        .select(['trade_time', 'code', "tc001_006"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
