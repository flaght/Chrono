"""
因子代号: tc001_029
原历史名: tc001_029
因子定义: 20周期价格相对位置 = (close - rolling_min(low, 20)) / (rolling_max(high, 20) - rolling_min(low, 20))
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算20周期价格位置因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_029 列
    """

    # 计算步骤:
    # 1. 计算20周期最高价和最低价
    # 2. 价格位置 = (收盘价 - 最低价) / (最高价 - 最低价)
    # 3. 取值范围[0, 1]，越接近1表示价格越接近区间上沿
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 20周期最高价
            pl.col('high').rolling_max(20).over('code').alias('high_20d'),
            # 20周期最低价
            pl.col('low').rolling_min(20).over('code').alias('low_20d')
        )
        .with_columns(
            # 价格位置 = (收盘价 - 最低价) / (最高价 - 最低价)
            # 当最高价=最低价时，设为0.5（中性值）
            pl.when(pl.col('high_20d') == pl.col('low_20d'))
            .then(0.5)
            .otherwise((pl.col('close') - pl.col('low_20d')) / 
                      (pl.col('high_20d') - pl.col('low_20d')))
            .alias("tc001_029")
        )
        .select(['trade_time', 'code', "tc001_029"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
