"""
因子代号: tc001_005
原历史名: tc001_005
因子定义: 14周期平均真实波幅 ATR = rolling_mean(TR, 14)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算14周期平均真实波幅ATR因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_005 列
    """

    # 计算步骤:
    # 1. 计算真实波幅 TR = max(最高价-最低价, |最高价-昨收|, |最低价-昨收|)
    # 2. 计算14周期TR的移动平均作为ATR
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 获取前一周期收盘价
            pl.col('close').shift(1).over('code').alias('pre_close')
        )
        .with_columns(
            # 真实波幅 TR = max(最高价-最低价, |最高价-昨收|, |最低价-昨收|)
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('pre_close')).abs(),
                (pl.col('low') - pl.col('pre_close')).abs()
            ]).alias('tr')
        )
        .with_columns(
            # 14周期平均真实波幅 ATR
            pl.col('tr').rolling_mean(14).over('code').alias("tc001_005")
        )
        .select(['trade_time', 'code', "tc001_005"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
