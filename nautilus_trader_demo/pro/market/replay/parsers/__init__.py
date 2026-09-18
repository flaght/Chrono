"""Normalized market-data parsers with compatibility-friendly lazy exports."""

from __future__ import annotations

from typing import Any


__all__ = [
    "BarColumns",
    "BinanceKlineParser",
    "BinanceMarketType",
    "CtpTickParser",
    "MappedBarParser",
    "binance_instrument_id",
]


def __getattr__(name: str) -> Any:
    if name == "CtpTickParser":
        from market.replay.parsers.tick import CtpTickParser

        return CtpTickParser
    if name in {
        "BarColumns",
        "BinanceKlineParser",
        "BinanceMarketType",
        "MappedBarParser",
        "binance_instrument_id",
    }:
        from market.replay.parsers.bar import (
            BarColumns,
            BinanceKlineParser,
            BinanceMarketType,
            MappedBarParser,
            binance_instrument_id,
        )

        return {
            "BarColumns": BarColumns,
            "BinanceKlineParser": BinanceKlineParser,
            "BinanceMarketType": BinanceMarketType,
            "MappedBarParser": MappedBarParser,
            "binance_instrument_id": binance_instrument_id,
        }[name]
    raise AttributeError(name)
