"""
因子代号: tc001_021
原历史名: tc001_021
因子定义: 确然指标 KST = 0.1*EMA(ROC10, 10) + 0.2*EMA(ROC15, 10) + 0.3*EMA(ROC20, 10) + 0.4*EMA(ROC30, 15)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算确然指标(KST)因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_021 列
    """

    # 计算步骤:
    # 1. 计算4个不同周期的ROC（变动率）
    # 2. 对每个ROC做EMA平滑
    # 3. 加权求和得到KST
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # ROC1 = close / close.shift(10) - 1
            (pl.col('close') / pl.col('close').shift(10).over('code') - 1).alias('roc1'),
            # ROC2 = close / close.shift(15) - 1
            (pl.col('close') / pl.col('close').shift(15).over('code') - 1).alias('roc2'),
            # ROC3 = close / close.shift(20) - 1
            (pl.col('close') / pl.col('close').shift(20).over('code') - 1).alias('roc3'),
            # ROC4 = close / close.shift(30) - 1
            (pl.col('close') / pl.col('close').shift(30).over('code') - 1).alias('roc4')
        )
        .with_columns(
            # 对每个ROC做EMA平滑
            pl.col('roc1').ewm_mean(span=10).over('code').alias('roc1_ema'),
            pl.col('roc2').ewm_mean(span=10).over('code').alias('roc2_ema'),
            pl.col('roc3').ewm_mean(span=10).over('code').alias('roc3_ema'),
            pl.col('roc4').ewm_mean(span=15).over('code').alias('roc4_ema')
        )
        .with_columns(
            # KST = (ROC1_EMA*1 + ROC2_EMA*2 + ROC3_EMA*3 + ROC4_EMA*4) / 10
            ((pl.col('roc1_ema') * 1
              + pl.col('roc2_ema') * 2
              + pl.col('roc3_ema') * 3
              + pl.col('roc4_ema') * 4) / 10).alias("tc001_021")
        )
        .select(['trade_time', 'code', "tc001_021"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
