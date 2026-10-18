"""
因子代号: tc001_030
原历史名: tc001_030
因子定义: 20周期价量相关系数 = rolling_corr(close, volume, 20)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算20周期价量相关系数因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_030 列
    """

    # 计算步骤:
    # 1. 计算20周期收盘价与成交量的滚动相关系数
    # 2. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 20周期价量相关系数
            pl.rolling_corr(
                pl.col('close'),
                pl.col('volume'),
                window_size=20
            ).over('code')
            .alias("tc001_030")
        )
        .select(['trade_time', 'code', "tc001_030"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
