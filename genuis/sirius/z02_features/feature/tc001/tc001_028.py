"""
因子代号: tc001_028
原历史名: tc001_028
因子定义: 百分比价格振荡器 PPO = (EMA(close, 12) - EMA(close, 26)) / EMA(close, 26) * 100
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算百分比价格振荡器(PPO)因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_028 列
    """

    # 计算步骤:
    # 1. 计算12周期EMA
    # 2. 计算26周期EMA
    # 3. PPO = (EMA12 - EMA26) / EMA26 * 100
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
            # PPO = (EMA12 - EMA26) / EMA26 * 100
            # 当EMA26=0时，设为0
            pl.when(pl.col('ema_26') == 0)
            .then(0.0)
            .otherwise((pl.col('ema_12') - pl.col('ema_26')) / pl.col('ema_26') * 100)
            .alias("tc001_028")
        )
        .select(['trade_time', 'code', "tc001_028"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
