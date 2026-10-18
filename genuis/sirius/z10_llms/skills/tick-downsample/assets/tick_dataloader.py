"""
CTP Tick 数据懒加载器 (Tick DataLoader)

负责从 Parquet、Feather (Arrow IPC) 或 CSV 中以 Lazy 模式加载原始 CTP Tick 流，
统一规整字段命名与标准 timestamp 列。
"""

from __future__ import annotations

from pathlib import Path
import polars as pl


REQUIRED_TICK_COLUMNS = [
    "TradingDay",
    "InstrumentID",
    "UpdateTime",
    "UpdateMillisec",
    "LastPrice",
    "Volume",
    "Turnover",
    "AveragePrice",
    "BidPrice1",
    "BidVolume1",
    "AskPrice1",
    "AskVolume1",
    "OpenInterest",
    "UpperLimitPrice",
    "LowerLimitPrice",
]


def load_tick_data(file_path: str | Path) -> pl.LazyFrame:
    """
    以纯 LazyFrame 方式加载原始 CTP Tick 数据并规整时间戳。

    参数:
        file_path: 数据文件路径 (支持 .parquet, .feather, .ipc, .csv)
    返回:
        pl.LazyFrame: 包含规范化 timestamp 列的 Tick 懒加载流
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tick 数据文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix in (".feather", ".ipc"):
        df_lazy = pl.scan_ipc(path)
    elif suffix == ".parquet":
        df_lazy = pl.scan_parquet(path)
    elif suffix == ".csv":
        df_lazy = pl.scan_csv(path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .parquet / .feather / .ipc / .csv")

    # 验证字段完备性
    schema = df_lazy.collect_schema()
    missing = [col for col in REQUIRED_TICK_COLUMNS if col not in schema.names()]
    if missing:
        raise ValueError(f"Tick 数据缺失必要字段: {missing}")

    # 若尚未包含标准 timestamp 列，则使用 TradingDay + UpdateTime + UpdateMillisec 合成
    if "timestamp" not in schema.names():
        df_lazy = df_lazy.with_columns(
            pl.concat_str([
                pl.col("TradingDay").cast(pl.Utf8),
                pl.lit(" "),
                pl.col("UpdateTime").cast(pl.Utf8),
                pl.lit("."),
                pl.col("UpdateMillisec").cast(pl.Utf8).str.zfill(3),
            ])
            .str.strptime(pl.Datetime, format="%Y%m%d %H:%M:%S.%3f", strict=False)
            .alias("timestamp")
        )

    return df_lazy
