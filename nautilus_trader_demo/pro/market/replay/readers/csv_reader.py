"""CSV row reader."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from market.replay.readers.base import ReaderError, require_supported_suffix


class CsvReader:
    suffixes = frozenset({".csv"})

    def __init__(self, encoding: str = "utf-8-sig") -> None:
        self.encoding = encoding

    def read(self, path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
        require_supported_suffix(path, self.suffixes)
        try:
            stream = path.open("r", encoding=self.encoding, newline="")
        except OSError as exc:
            raise ReaderError(f"cannot open CSV file: {path}: {exc}") from exc

        with stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ReaderError(f"CSV file has no header: {path}")
            for line, row in enumerate(reader, start=2):
                yield line, row
