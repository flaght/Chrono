"""
因子代号: tc001_020
原历史名: tc001_020
因子定义: 9周期KDJ随机指标 K值 = rolling_mean((close - low_9) / (high_9 - low_9) * 100, 3)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算9周期KDJ因子（K值）

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_020 列
    """

    # 计算步骤:
    # 1. 计算9单周期最高价和最低价
    # 2. 计算RSV = 100 * (收盘价 - 最低价) / (最高价 - 最低价)
    # 3. K值 = RSV的3周期移动平均
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 9单周期最高价
            pl.col('high').rolling_max(9).over('code').alias('high_9'),
            # 9单周期最低价
            pl.col('low').rolling_min(9).over('code').alias('low_9')
        )
        .with_columns(
            # RSV = 100 * (收盘价 - 最低价) / (最高价 - 最低价)
            # 当最高价=最低价时，RSV设为50（中性值）
            pl.when(pl.col('high_9') == pl.col('low_9'))
            .then(50.0)
            .otherwise(100.0 * (pl.col('close') - pl.col('low_9')) / (pl.col('high_9') - pl.col('low_9')))
            .alias('rsv')
        )
        .with_columns(
            # K值 = RSV的3周期移动平均
            pl.col('rsv').rolling_mean(3).over('code').alias("tc001_020")
        )
        .select(['trade_time', 'code', "tc001_020"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
