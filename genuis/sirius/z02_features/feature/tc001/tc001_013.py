"""
因子代号: tc001_013
原历史名: tc001_013
因子定义: 重心指标 CG = -sum(close_i * i, 10) / sum(close_i, 10)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算重心指标(Center of Gravity)因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_013 列
    """

    # 计算步骤:
    # 1. 对每周期生成10个滞后价格，权重分别为1到10
    # 2. CG = -sum(price_i * i) / sum(price_i)
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 生成10个滞后收盘价，权重为时间位置1到10
            # 分子 = sum(close_i * i)，i=1..10
            (pl.col('close').shift(0).over('code') * 1
             + pl.col('close').shift(1).over('code') * 2
             + pl.col('close').shift(2).over('code') * 3
             + pl.col('close').shift(3).over('code') * 4
             + pl.col('close').shift(4).over('code') * 5
             + pl.col('close').shift(5).over('code') * 6
             + pl.col('close').shift(6).over('code') * 7
             + pl.col('close').shift(7).over('code') * 8
             + pl.col('close').shift(8).over('code') * 9
             + pl.col('close').shift(9).over('code') * 10
             ).alias('numerator'),
            # 分母 = sum(close_i)，i=1..10
            (pl.col('close').shift(0).over('code')
             + pl.col('close').shift(1).over('code')
             + pl.col('close').shift(2).over('code')
             + pl.col('close').shift(3).over('code')
             + pl.col('close').shift(4).over('code')
             + pl.col('close').shift(5).over('code')
             + pl.col('close').shift(6).over('code')
             + pl.col('close').shift(7).over('code')
             + pl.col('close').shift(8).over('code')
             + pl.col('close').shift(9).over('code')
             ).alias('denominator')
        )
        .with_columns(
            # CG = -分子 / 分母
            # 当分母=0时，设为0
            pl.when(pl.col('denominator') == 0)
            .then(0.0)
            .otherwise(-pl.col('numerator') / pl.col('denominator'))
            .alias("tc001_013")
        )
        .select(['trade_time', 'code', "tc001_013"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
