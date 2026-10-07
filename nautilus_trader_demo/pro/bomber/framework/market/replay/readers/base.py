"""离线行情的存储格式读取器契约。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


class ReaderError(ValueError):
    """逐行读取器无法打开或解码文件。"""


class RowReader(Protocol):
    """读取存储行，不解释行情业务语义。"""

    suffixes: frozenset[str]

    def read(self, path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]: ...


def require_supported_suffix(path: Path, suffixes: frozenset[str]) -> None:
    if path.suffix.lower() not in suffixes:
        expected = ", ".join(sorted(suffixes))
        raise ReaderError(f"expected one of [{expected}], got: {path}")
