"""Apache Feather row reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from market.replay.readers.base import ReaderError, require_supported_suffix


class FeatherReader:
    suffixes = frozenset({".feather"})

    def read(self, path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
        require_supported_suffix(path, self.suffixes)
        try:
            import pyarrow.feather as feather
        except ImportError as exc:
            raise ReaderError("reading Feather files requires pyarrow") from exc
        try:
            table = feather.read_table(path)
        except Exception as exc:
            raise ReaderError(f"cannot read Feather file: {path}: {exc}") from exc
        for row_number, row in enumerate(table.to_pylist(), start=1):
            yield row_number, row
