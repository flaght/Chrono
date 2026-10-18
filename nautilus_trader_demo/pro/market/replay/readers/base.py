"""Storage-format reader contracts for offline market data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


class ReaderError(ValueError):
    """A file cannot be opened or decoded by a row reader."""


class RowReader(Protocol):
    """Read storage rows without interpreting market-data semantics."""

    suffixes: frozenset[str]

    def read(self, path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]: ...


def require_supported_suffix(path: Path, suffixes: frozenset[str]) -> None:
    if path.suffix.lower() not in suffixes:
        expected = ", ".join(sorted(suffixes))
        raise ReaderError(f"expected one of [{expected}], got: {path}")
