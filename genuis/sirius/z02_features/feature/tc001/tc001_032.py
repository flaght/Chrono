"""
因子代号: tc001_032
原历史名: tc001_032
因子定义: 百分比成交量振荡器 PVO = (EMA(volume, 12) - EMA(volume, 26)) / EMA(volume, 26) * 100
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算百分比成交量振荡器(PVO)因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_032 列
    """

    # 计算步骤:
    # 1. 计算成交量的12周期EMA
    # 2. 计算成交量的26周期EMA
    # 3. PVO = (EMA12 - EMA26) / EMA26 * 100
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 成交量12周期EMA
            pl.col('volume').ewm_mean(span=12).over('code').alias('vol_ema_12'),
            # 成交量26周期EMA
            pl.col('volume').ewm_mean(span=26).over('code').alias('vol_ema_26')
        )
        .with_columns(
            # PVO = (EMA12 - EMA26) / EMA26 * 100
            # 当EMA26=0时，设为0
            pl.when(pl.col('vol_ema_26') == 0)
            .then(0.0)
            .otherwise(
                (pl.col('vol_ema_12') - pl.col('vol_ema_26')) / pl.col('vol_ema_26') * 100
            )
            .alias("tc001_032")
        )
        .select(['trade_time', 'code', "tc001_032"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
