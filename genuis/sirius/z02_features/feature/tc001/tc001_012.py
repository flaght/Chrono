"""
因子代号: tc001_012
原历史名: tc001_012
因子定义: 14周期顺势指标 CCI = (TP - SMA(TP, 14)) / (0.015 * MD(TP, 14))
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算14周期CCI因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_012 列
    """

    # 计算步骤:
    # 1. 计算典型价格 TP = (最高价 + 最低价 + 收盘价) / 3
    # 2. 计算14周期TP的移动平均
    # 3. 计算14周期TP的平均绝对偏差
    # 4. CCI = (TP - TP均值) / (0.015 * 平均绝对偏差)
    # 5. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 典型价格 TP = (最高价 + 最低价 + 收盘价) / 3
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3)
            .alias('tp')
        )
        .with_columns(
            # 14周期TP的移动平均
            pl.col('tp').rolling_mean(14).over('code').alias('tp_mean')
        )
        .with_columns(
            # 当前TP相对14周期均值的绝对偏差
            (pl.col('tp') - pl.col('tp_mean')).abs().alias('tp_deviation')
        )
        .with_columns(
            # 14周期TP的平均绝对偏差
            pl.col('tp_deviation')
            .rolling_mean(14)
            .over('code')
            .alias('tp_mad')
        )
        .with_columns(
            # CCI = (TP - TP均值) / (0.015 * 平均绝对偏差)
            # 当平均绝对偏差=0时，CCI设为0
            pl.when(pl.col('tp_mad') == 0)
            .then(0.0)
            .otherwise((pl.col('tp') - pl.col('tp_mean')) / (0.015 * pl.col('tp_mad')))
            .alias("tc001_012")
        )
        .select(['trade_time', 'code', "tc001_012"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
