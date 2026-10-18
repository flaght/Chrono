"""
因子代号: tc001_031
原历史名: tc001_031
因子定义: 12周期心理线 PSL = count(close >= open, 12) / 12 * 100
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算心理信号线(PSL)因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、open
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_031 列
    """

    # 计算步骤:
    # 1. 标记收盘价>=开盘价的日期（阳线）
    # 2. 计算12周期内上涨周期数占比
    # 3. PSL = 上涨周期数 / 12 * 100
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 标记阳线日：收盘价 >= 开盘价
            pl.when(pl.col('close') >= pl.col('open'))
            .then(1.0)
            .otherwise(0.0)
            .alias('is_positive')
        )
        .with_columns(
            # 12周期内上涨周期数
            pl.col('is_positive').rolling_sum(12).over('code').alias('positive_count')
        )
        .with_columns(
            # PSL = 上涨周期数 / 12 * 100
            (pl.col('positive_count') / 12 * 100).alias("tc001_031")
        )
        .select(['trade_time', 'code', "tc001_031"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
