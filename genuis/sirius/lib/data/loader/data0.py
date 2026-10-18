"""BTC 数据准备与聚合模块。

支持从 z21_orchestrator 的基础数据目录 (futures / spot) 懒式扫描并对齐，
生成包含 future_close 和 spot_close 的 LazyFrame，供 four_ma 因子使用。
"""

import os
import glob
import polars as pl


def load_returns(returns_file):
    filename = os.path.join(returns_file)
    df_lazy = pl.scan_ipc(filename)
    return df_lazy

### 期货现货合并数据
def load_basis_lazy(
    futures_dir: str,
    spot_dir: str,
    code: str = "BTCUSDT",
) -> pl.LazyFrame:
    """懒式加载期货与现货 CSV 数据，并按 trade_time 和 code 精确对齐。

    参数:
        futures_dir: 期货 CSV 目录
        spot_dir: 现货 CSV 目录
        code: 标的代码，默认为 BTCUSDT
    返回:
        pl.LazyFrame: 包含 trade_time, code, future_close, spot_close 的对齐计算图
    """
    futures_files = sorted(glob.glob(os.path.join(futures_dir, "*.csv")))
    spot_files = sorted(glob.glob(os.path.join(spot_dir, "*.csv")))

    if not futures_files:
        raise FileNotFoundError(f"未找到期货 CSV: {futures_dir}")
    if not spot_files:
        raise FileNotFoundError(f"未找到现货 CSV: {spot_dir}")

    # 1. 懒式扫描期货
    future_lazy = (
        pl.scan_csv(futures_files)
        .with_columns([
            pl.lit(code).alias("code"),
            pl.col("trade_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        ])
        .select([
            "trade_time",
            "code",
            pl.col("close").alias("future_close"),
            pl.col("open").alias("future_open"),
            pl.col("high").alias("future_high"),
            pl.col("low").alias("future_low"),
            pl.col("volume").alias("future_volume"),
        ])
    )

    # 2. 懒式扫描现货
    spot_lazy = (
        pl.scan_csv(spot_files)
        .with_columns([
            pl.lit(code).alias("code"),
            pl.col("trade_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        ])
        .select([
            "trade_time",
            "code",
            pl.col("close").alias("spot_close"),
            pl.col("open").alias("spot_open"),
            pl.col("high").alias("spot_high"),
            pl.col("low").alias("spot_low"),
            pl.col("volume").alias("spot_volume"),
        ])
    )

    # 3. 按 trade_time 和 code 精确连接
    joined_lazy = (
        future_lazy
        .join(spot_lazy, on=["trade_time", "code"], how="inner")
        .sort(["trade_time", "code"])
    )

    return joined_lazy


def load_data_lazy(data_dir:str, code: str = "BTCUSDT"):
    data_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not data_files:
        raise FileNotFoundError(f"未找到期货 CSV: {data_dir}")

    data_lazy = (
        pl.scan_csv(data_files)
        .with_columns([
            pl.lit(code).alias("code"),
            pl.col("trade_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        ])
        .select([
            "trade_time",
            "code",
            pl.col("close"),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("volume"),
            pl.col('quote_volume').alias('value')
        ])
    )
    return data_lazy