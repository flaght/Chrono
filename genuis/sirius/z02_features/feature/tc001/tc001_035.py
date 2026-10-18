"""
因子代号: tc001_035
原历史名: tc001_035
因子定义: 20周期成交额标准差 = rolling_std(value, 20)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算20周期成交额波动率因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、value
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_035 列
    """

    # 计算步骤:
    # 1. 计算20周期成交额的滚动标准差
    # 2. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .with_columns(
            # 20周期成交额标准差
            pl.col('value')
            .rolling_std(20)
            .over('code')
            .alias("tc001_035")
        )
        .select(['trade_time', 'code', "tc001_035"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
