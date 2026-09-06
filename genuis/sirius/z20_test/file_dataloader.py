"""从指定文件懒加载基础行情数据，并原样保留字段。"""

from pathlib import Path
from typing import Union

import polars as pl


COMMON_COLUMNS = (
    "trade_time",
    "code",
    "high",
    "low",
    "open",
    "close",
    "volume",
    "value",
)


def dataloader(file_path: Union[str, Path]) -> pl.LazyFrame:
    """
    懒加载指定的 Feather/Arrow IPC 或 Parquet 文件。

    参数:
        file_path: 单个 .feather、.arrow、.ipc 或 .parquet 文件
    返回:
        pl.LazyFrame: 原样保留文件中的字段名和字段类型
    """
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix in {".feather", ".arrow", ".ipc"}:
        df_lazy = pl.scan_ipc(path)
    elif suffix == ".parquet":
        df_lazy = pl.scan_parquet(path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    schema_names = set(df_lazy.collect_schema().names())
    missing_columns = [name for name in COMMON_COLUMNS if name not in schema_names]
    if missing_columns:
        raise ValueError(f"基础行情文件缺少字段: {missing_columns}")

    return df_lazy



file_path = "/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/basic/train_data.feather"

df_lazy = dataloader(file_path)
print(df_lazy.collect())