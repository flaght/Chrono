"""
因子代号: tf002_013
原历史名: tf002_013
因子定义: N期收盘价对数收益率、最高价极差、持仓量变化率三者的三阶混合分布熵复合因子，衡量收益、极端波动与持仓变化的高阶不确定性
"""
import polars as pl

from feature.utils.common import (
    rolling_rank,
    rolling_sign_change_rate,
    validate_period,
)

DEFAULT_PERIOD = 15
NAME = "tf002_013"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """使用trade_time、code、close、high、low、openint计算 tf002_013 核心公式。"""
    return (
        df_lazy
        .with_columns(
            ((pl.col('close')).shift(1).over("code")).alias("_cr061_stage_0")
        )
        .with_columns(
            (pl.when((pl.col("_cr061_stage_0")).is_not_null() & ((pl.col("_cr061_stage_0")) != 0)).then((pl.col('close')) / (pl.col("_cr061_stage_0"))).otherwise(None)).alias("_cr061_stage_1")
        )
        .with_columns(
            (pl.when((pl.col("_cr061_stage_1")) > 0).then((pl.col("_cr061_stage_1")).log()).otherwise(None)).alias("_cr061_stage_2")
        )
        .with_columns(
            ((pl.col("_cr061_stage_2")).rolling_mean(period).over("code")).alias("_cr061_stage_3")
        )
        .with_columns(
            ((pl.col("_cr061_stage_2")) - (pl.col("_cr061_stage_3"))).alias("_cr061_stage_4")
        )
        .with_columns(
            ((pl.col('high')).rolling_max(period).over("code")).alias("_cr061_stage_5")
        )
        .with_columns(
            ((pl.col('low')).rolling_min(period).over("code")).alias("_cr061_stage_6")
        )
        .with_columns(
            ((pl.col("_cr061_stage_5")) - (pl.col("_cr061_stage_6"))).alias("_cr061_stage_7")
        )
        .with_columns(
            ((pl.col("_cr061_stage_7")).rolling_mean(period).over("code")).alias("_cr061_stage_8")
        )
        .with_columns(
            ((pl.col("_cr061_stage_7")) - (pl.col("_cr061_stage_8"))).alias("_cr061_stage_9")
        )
        .with_columns(
            ((pl.col("_cr061_stage_4")) * (pl.col("_cr061_stage_9"))).alias("_cr061_stage_10")
        )
        .with_columns(
            ((pl.col('openint')).pct_change().over("code")).alias("_cr061_stage_11")
        )
        .with_columns(
            ((pl.col("_cr061_stage_11")).rolling_mean(period).over("code")).alias("_cr061_stage_12")
        )
        .with_columns(
            ((pl.col("_cr061_stage_11")) - (pl.col("_cr061_stage_12"))).alias("_cr061_stage_13")
        )
        .with_columns(
            ((pl.col("_cr061_stage_10")) * (pl.col("_cr061_stage_13"))).alias("_cr061_stage_14")
        )
        .with_columns(
            ((pl.col("_cr061_stage_14")).rolling_min(period).over("code")).alias("_cr061_stage_15")
        )
        .with_columns(
            ((pl.col("_cr061_stage_14")) - (pl.col("_cr061_stage_15"))).alias("_cr061_stage_16")
        )
        .with_columns(
            ((pl.col("_cr061_stage_14")).rolling_max(period).over("code")).alias("_cr061_stage_17")
        )
        .with_columns(
            ((pl.col("_cr061_stage_17")) - (pl.col("_cr061_stage_15"))).alias("_cr061_stage_18")
        )
        .with_columns(
            (pl.when((pl.col("_cr061_stage_18")).is_not_null() & ((pl.col("_cr061_stage_18")) != 0)).then((pl.col("_cr061_stage_16")) / (pl.col("_cr061_stage_18"))).otherwise(None)).alias("_cr061_stage_19")
        )
        .with_columns(
            ((pl.col("_cr061_stage_19")).clip(1e-08, 1.0)).alias("_cr061_stage_20")
        )
        .with_columns(
            (pl.when((pl.col("_cr061_stage_20")) > 0).then((pl.col("_cr061_stage_20")).log()).otherwise(None)).alias("_cr061_stage_21")
        )
        .with_columns(
            ((pl.col("_cr061_stage_20")) * (pl.col("_cr061_stage_21"))).alias("_cr061_stage_22")
        )
        .with_columns(
            ((pl.col("_cr061_stage_22")).rolling_sum(period).over("code")).alias("_cr061_stage_23")
        )
        .with_columns(
            (-(pl.col("_cr061_stage_23"))).alias("_cr061_stage_24")
        )
        .with_columns(
            (pl.col("_cr061_stage_24")).alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(df_lazy: pl.LazyFrame, period: int = DEFAULT_PERIOD) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf002_013；输入必须包含trade_time、code、close、high、low、openint。"""
    validate_period(period)
    return calculate(df_lazy.sort(["trade_time", "code"]), period).select(["trade_time", "code", NAME])

