"""
因子代号: tc001_019
原历史名: tc001_019
因子定义: 20周期日内收益均值 = rolling_mean(close / open - 1, 20)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算20周期单周期收益率均值因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、open
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_019 列
    """

    # 计算步骤:
    # 1. 计算单周期收益率 = 收盘价 / 开盘价 - 1
    # 2. 计算20周期单周期收益率均值
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 单周期收益率 = 收盘价 / 开盘价 - 1
            (pl.col('close') / pl.col('open') - 1).alias('intraday_ret')
        )
        .with_columns(
            # 20周期单周期收益率均值
            pl.col('intraday_ret').rolling_mean(20).over('code').alias("tc001_019")
        )
        .select(['trade_time', 'code', "tc001_019"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
