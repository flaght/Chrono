"""
因子代号: tc001_037
原历史名: tc001_037
因子定义: 20周期量比 = rolling_mean(volume, 5) / rolling_mean(volume, 20)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算20周期量比因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_037 列
    """

    # 计算步骤:
    # 1. 计算5周期平均成交量
    # 2. 计算20周期平均成交量
    # 3. 量比 = 5周期均量 / 20周期均量
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .with_columns(
            # 5周期平均成交量
            pl.col('volume').rolling_mean(5).over('code').alias('vol_5d'),
            # 20周期平均成交量
            pl.col('volume').rolling_mean(20).over('code').alias('vol_20d')
        )
        .with_columns(
            # 量比 = 5周期均量 / 20周期均量
            # 当20周期均量=0时，设为1（中性值）
            pl.when(pl.col('vol_20d') == 0)
            .then(1.0)
            .otherwise(pl.col('vol_5d') / pl.col('vol_20d'))
            .alias("tc001_037")
        )
        .select(['trade_time', 'code', "tc001_037"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
