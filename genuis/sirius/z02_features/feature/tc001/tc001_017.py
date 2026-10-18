"""
因子代号: tc001_017
原历史名: tc001_017
因子定义: 52周期最高价距离 = close / rolling_max(close, 52) - 1
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算250周期最高价距离因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_017 列
    """

    # 计算步骤:
    # 1. 计算250周期最高收盘价
    # 2. 250周期最高价距离 = 收盘价 / 250周期最高价 - 1
    # 3. 取值范围[-1, 0]，越接近0表示越接近250周期新高
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 250周期最高收盘价
            pl.col('close').rolling_max(250).over('code').alias('high_250d')
        )
        .with_columns(
            # 250周期最高价距离 = 收盘价 / 250周期最高价 - 1
            # 当250周期最高价=0时，设为0
            pl.when(pl.col('high_250d') == 0)
            .then(0.0)
            .otherwise(pl.col('close') / pl.col('high_250d') - 1)
            .alias("tc001_017")
        )
        .select(['trade_time', 'code', "tc001_017"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
