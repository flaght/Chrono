"""
因子代号: tc001_004
原历史名: tc001_004
因子定义: 绝对价格振荡器 APO = EMA(close, 12) - EMA(close, 26)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算绝对价格振荡器(APO)因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_004 列
    """

    # 计算步骤:
    # 1. 计算12周期指数移动平均线 EMA12
    # 2. 计算26周期指数移动平均线 EMA26
    # 3. APO = EMA12 - EMA26
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 12周期指数移动平均线
            pl.col('close').ewm_mean(span=12).over('code').alias('ema_12'),
            # 26周期指数移动平均线
            pl.col('close').ewm_mean(span=26).over('code').alias('ema_26')
        )
        .with_columns(
            # APO = EMA12 - EMA26
            (pl.col('ema_12') - pl.col('ema_26')).alias("tc001_004")
        )
        .select(['trade_time', 'code', "tc001_004"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
