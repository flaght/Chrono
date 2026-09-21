"""Real-time market-data feed packages.

Adapters are imported lazily so that an unavailable or incomplete provider does
not prevent another provider package from being used.
"""

from __future__ import annotations

from typing import Any

from market.stream.base import StreamDataFeed
from market.stream.health import (
    MarketHealthReason,
    MarketHealthSnapshot,
    MarketHealthState,
    StreamHealthConfig,
)
from market.stream.aggregation import QuoteMidBarFeed, TradeTickBarFeed

__all__ = [
    "StreamDataFeed",
    "MarketHealthReason",
    "MarketHealthSnapshot",
    "MarketHealthState",
    "StreamHealthConfig",
    "TradeTickBarFeed",
    "QuoteMidBarFeed",
    "BNWSConfig",
    "BNWSStreamDataFeed",
    "CtpMdConfig",
    "CtpLiveDataFeed",
    "CtpTickConverter",
    "DolphinDbConfig",
    "DolphinDbLiveDataFeed",
    "DolphinDbMarketConverter",
    "DolphinDbStreamSpec",
]


def __getattr__(name: str) -> Any:
    if name in {"BNWSConfig", "BNWSStreamDataFeed"}:
        from market.stream.bn import BNWSConfig, BNWSStreamDataFeed

        return {
            "BNWSConfig": BNWSConfig,
            "BNWSStreamDataFeed": BNWSStreamDataFeed,
        }[name]

    if name in {"CtpMdConfig", "CtpLiveDataFeed", "CtpTickConverter"}:
        from market.stream.ctp import CtpLiveDataFeed, CtpMdConfig, CtpTickConverter

        return {
            "CtpMdConfig": CtpMdConfig,
            "CtpLiveDataFeed": CtpLiveDataFeed,
            "CtpTickConverter": CtpTickConverter,
        }[name]

    if name in {
        "DolphinDbConfig",
        "DolphinDbLiveDataFeed",
        "DolphinDbMarketConverter",
        "DolphinDbStreamSpec",
    }:
        from market.stream.dolphin import (
            DolphinDbConfig,
            DolphinDbLiveDataFeed,
            DolphinDbMarketConverter,
            DolphinDbStreamSpec,
        )

        return {
            "DolphinDbConfig": DolphinDbConfig,
            "DolphinDbLiveDataFeed": DolphinDbLiveDataFeed,
            "DolphinDbMarketConverter": DolphinDbMarketConverter,
            "DolphinDbStreamSpec": DolphinDbStreamSpec,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
