"""懒加载期货和现货行情，并准备基差因子输入字段。"""

from pathlib import Path
from typing import Union

import polars as pl

try:
    from .file_dataloader import dataloader
except ImportError:
    from file_dataloader import dataloader


PathLike = Union[str, Path]
FUTURE_REQUIRED_COLUMNS = {
    "trade_time", "code", "open", "high", "low", "close",
    "volume", "value", "openint",
}
SPOT_REQUIRED_COLUMNS = {
    "trade_time", "code", "open", "high", "low", "close",
    "volume", "value",
}


def _require_columns(
    df_lazy: pl.LazyFrame,
    required_columns: set,
    source_name: str,
) -> None:
    schema_names = set(df_lazy.collect_schema().names())
    missing_columns = sorted(required_columns - schema_names)
    if missing_columns:
        raise ValueError(f"{source_name}数据缺少字段: {missing_columns}")


def load_basis_data(
    future_file_path: PathLike,
    spot_file_path: PathLike,
) -> pl.LazyFrame:
    """
    懒加载期货与现货文件，添加前缀并按时间和标的精确对齐。

    参数:
        future_file_path: 期货 Feather/IPC 或 Parquet 文件
        spot_file_path: 现货 Feather/IPC 或 Parquet 文件
    返回:
        pl.LazyFrame: 包含 trade_time、code、future_* 和 spot_* 字段

    约束:
        期货和现货使用相同 code 表示同一品种，例如都为 RB。
    """
    future_lazy = dataloader(future_file_path)
    spot_lazy = dataloader(spot_file_path)

    _require_columns(
        future_lazy,
        FUTURE_REQUIRED_COLUMNS,
        "期货",
    )
    _require_columns(
        spot_lazy,
        SPOT_REQUIRED_COLUMNS,
        "现货",
    )

    future_schema = future_lazy.collect_schema()
    spot_schema = spot_lazy.collect_schema()
    if future_schema["trade_time"] != spot_schema["trade_time"]:
        raise TypeError(
            "期货和现货 trade_time 类型必须一致: "
            f"{future_schema['trade_time']} != {spot_schema['trade_time']}"
        )

    future_prepared = future_lazy.select(
        pl.col("trade_time"),
        pl.col("code"),
        pl.col("open").alias("future_open"),
        pl.col("high").alias("future_high"),
        pl.col("low").alias("future_low"),
        pl.col("close").alias("future_close"),
        pl.col("volume").alias("future_volume"),
        pl.col("value").alias("future_value"),
        pl.col("openint").alias("future_openint"),
    )

    spot_prepared = spot_lazy.select(
        pl.col("trade_time"),
        pl.col("code"),
        pl.col("open").alias("spot_open"),
        pl.col("high").alias("spot_high"),
        pl.col("low").alias("spot_low"),
        pl.col("close").alias("spot_close"),
        pl.col("volume").alias("spot_volume"),
        pl.col("value").alias("spot_value"),
    )

    return (
        future_prepared
        .join(
            spot_prepared,
            on=["trade_time", "code"],
            how="inner",
            validate="m:1",
        )
        .sort(by=["trade_time", "code"])
    )
