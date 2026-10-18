"""
因子代号: tc001_002
原历史名: tc001_002
因子定义: 累积/派发线 (AD Line) 20周期变化量 = rolling_sum(AD, 20)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算累积/派发线AD因子（20周期变化量）

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_002 列
    """

    # 计算步骤:
    # 1. 计算每周期AD值: 成交量 * (2*收盘价 - 最高价 - 最低价) / (最高价 + 最低价)
    # 2. 计算20周期AD的累积变化量
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 每周期AD值 = 成交量 * (2*收盘价 - 最高价 - 最低价) / (最高价 + 最低价)
            # 当最高价+最低价=0时，AD设为0
            pl.when(pl.col('high') + pl.col('low') == 0)
            .then(0.0)
            .otherwise(pl.col('volume') * 
                      (2 * pl.col('close') - pl.col('high') - pl.col('low')) / 
                      (pl.col('high') + pl.col('low')))
            .alias('daily_ad')
        )
        .with_columns(
            # 20周期AD累积变化量 = 20周期AD的滚动求和
            pl.col('daily_ad').rolling_sum(20).over('code').alias("tc001_002")
        )
        .select(['trade_time', 'code', "tc001_002"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
