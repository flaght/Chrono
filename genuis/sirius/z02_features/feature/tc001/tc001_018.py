"""
因子代号: tc001_018
原历史名: tc001_018
因子定义: 20周期高低价比率 = rolling_mean(high / low - 1, 20)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算20周期振幅均值因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_018 列
    """

    # 计算步骤:
    # 1. 计算每周期振幅 = 最高价 / 最低价 - 1
    # 2. 计算20周期振幅均值
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 每周期振幅 = 最高价 / 最低价 - 1
            # 当最低价=0时，设为0
            pl.when(pl.col('low') == 0)
            .then(0.0)
            .otherwise(pl.col('high') / pl.col('low') - 1)
            .alias('daily_amplitude')
        )
        .with_columns(
            # 20周期振幅均值
            pl.col('daily_amplitude').rolling_mean(20).over('code').alias("tc001_018")
        )
        .select(['trade_time', 'code', "tc001_018"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
