"""
因子代号: tc001_034
原历史名: tc001_034
因子定义: 12周期变动率 ROC = (close - close.shift(12)) / close.shift(12) * 100
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算12周期变动速率因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_034 列
    """

    # 计算步骤:
    # 1. 获取12周期前收盘价
    # 2. ROC = (当前周期收盘价 - 12周期前收盘价) / 12周期前收盘价 * 100
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 12周期前收盘价
            pl.col('close').shift(12).over('code').alias('close_12d_ago')
        )
        .with_columns(
            # ROC = (当前周期收盘价 - 12周期前收盘价) / 12周期前收盘价 * 100
            ((pl.col('close') - pl.col('close_12d_ago')) / pl.col('close_12d_ago') * 100)
            .alias("tc001_034")
        )
        .select(['trade_time', 'code', "tc001_034"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
