"""
因子代号: tf002_015
原历史名: tf002_015
因子定义: 短均线上穿长均线（金叉）事件，并结合持仓量上升与成交量下降构造量仓共振强度，经截面缩尾与 z-score 标准化。
"""
import polars as pl

DEFAULT_SHORT_MA_WINDOW = 5
DEFAULT_LONG_MA_WINDOW = 20
DEFAULT_OI_LOOKBACK = 10
DEFAULT_VOLUME_LOOKBACK = 10
DEFAULT_WINSORIZE_QUANTILE = 0.01
NAME = "tf002_015"


def calculate(
    df_lazy: pl.LazyFrame,
    short_ma_window: int,
    long_ma_window: int,
    oi_lookback: int,
    volume_lookback: int,
    winsorize_quantile: float,
) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 使用 trade_time、code、close、openint、volume 计算 tf002_015。"""
    return (
        df_lazy
        .with_columns([
            pl.col("close").rolling_mean(window_size=short_ma_window, min_periods=short_ma_window).over("code").alias("_ma_short"),
            pl.col("close").rolling_mean(window_size=long_ma_window, min_periods=long_ma_window).over("code").alias("_ma_long"),
            pl.col("openint").shift(1).over("code").alias("_oi_lag1"),
            pl.col("volume").shift(1).over("code").alias("_vol_lag1"),
        ])
        .with_columns([
            pl.col("_oi_lag1").rolling_mean(window_size=oi_lookback, min_periods=oi_lookback).over("code").alias("_oi_mean_prev"),
            pl.col("_vol_lag1").rolling_mean(window_size=volume_lookback, min_periods=volume_lookback).over("code").alias("_vol_mean_prev"),
        ])
        .with_columns([
            pl.col("_ma_short").shift(1).over("code").alias("_ma_short_prev"),
            pl.col("_ma_long").shift(1).over("code").alias("_ma_long_prev"),
            pl.when((pl.col("_oi_mean_prev") > 0) & (pl.col("openint") > 0))
            .then(pl.col("openint") / pl.col("_oi_mean_prev") - 1.0)
            .otherwise(None)
            .alias("_oi_change"),
            pl.when(pl.col("_vol_mean_prev") > 0)
            .then(pl.col("volume") / pl.col("_vol_mean_prev") - 1.0)
            .otherwise(None)
            .alias("_vol_change"),
        ])
        .with_columns(
            ((pl.col("_ma_short") > pl.col("_ma_long")) & (pl.col("_ma_short_prev") <= pl.col("_ma_long_prev"))).alias("_golden_cross")
        )
        .with_columns(
            pl.when(
                pl.col("_golden_cross") & (pl.col("_oi_change") > 0) & (pl.col("_vol_change") < 0)
            )
            .then(pl.col("_oi_change") * (-pl.col("_vol_change")))
            .otherwise(0.0)
            .alias("_raw")
        )
        .with_columns([
            pl.col("_raw").quantile(winsorize_quantile).over("trade_time").alias("_raw_q_low"),
            pl.col("_raw").quantile(1.0 - winsorize_quantile).over("trade_time").alias("_raw_q_high"),
        ])
        .with_columns(
            pl.when(
                pl.col("_raw").is_not_null() & (pl.col("_raw") < pl.col("_raw_q_low"))
            )
            .then(pl.col("_raw_q_low"))
            .when(
                pl.col("_raw").is_not_null() & (pl.col("_raw") > pl.col("_raw_q_high"))
            )
            .then(pl.col("_raw_q_high"))
            .otherwise(pl.col("_raw"))
            .alias("_raw_clipped")
        )
        .with_columns([
            pl.col("_raw_clipped").mean().over("trade_time").alias("_raw_mean"),
            pl.col("_raw_clipped").std(ddof=0).over("trade_time").alias("_raw_std"),
        ])
        .with_columns(
            pl.when(
                pl.col("_raw_clipped").is_not_null()
                & pl.col("_raw_std").is_not_null()
                & (pl.col("_raw_std") == 0)
            )
            .then(0.0)
            .otherwise(
                pl.when(
                    pl.col("_raw_clipped").is_not_null() & pl.col("_raw_std").is_not_null()
                )
                .then((pl.col("_raw_clipped") - pl.col("_raw_mean")) / pl.col("_raw_std"))
                .otherwise(None)
            )
            .alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(
    df_lazy: pl.LazyFrame,
    short_ma_window: int = DEFAULT_SHORT_MA_WINDOW,
    long_ma_window: int = DEFAULT_LONG_MA_WINDOW,
    oi_lookback: int = DEFAULT_OI_LOOKBACK,
    volume_lookback: int = DEFAULT_VOLUME_LOOKBACK,
    winsorize_quantile: float = DEFAULT_WINSORIZE_QUANTILE,
) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tf002_015；输入必须包含 trade_time、code、close、openint、volume。"""
    if not isinstance(short_ma_window, int) or isinstance(short_ma_window, bool) or short_ma_window <= 0:
        raise ValueError("short_ma_window 必须是正整数")
    if not isinstance(long_ma_window, int) or isinstance(long_ma_window, bool) or long_ma_window <= 0:
        raise ValueError("long_ma_window 必须是正整数")
    if long_ma_window <= short_ma_window:
        raise ValueError("long_ma_window 必须大于 short_ma_window")
    if not isinstance(oi_lookback, int) or isinstance(oi_lookback, bool) or oi_lookback <= 0:
        raise ValueError("oi_lookback 必须是正整数")
    if not isinstance(volume_lookback, int) or isinstance(volume_lookback, bool) or volume_lookback <= 0:
        raise ValueError("volume_lookback 必须是正整数")
    if not isinstance(winsorize_quantile, (int, float)) or isinstance(winsorize_quantile, bool):
        raise ValueError("winsorize_quantile 必须是数值")
    if not 0.0 <= winsorize_quantile < 0.5:
        raise ValueError("winsorize_quantile 必须在 [0, 0.5) 区间内")
    return calculate(
        df_lazy.sort(["trade_time", "code"]),
        short_ma_window,
        long_ma_window,
        oi_lookback,
        volume_lookback,
        winsorize_quantile,
    )