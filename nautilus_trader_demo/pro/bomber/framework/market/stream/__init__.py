"""实时行情源包。

适配器采用延迟导入，避免某个供应商不可用或安装不完整时
影响其他供应商模块的使用。
"""

from __future__ import annotations

from typing import Any

from bomber.framework.market.stream.base import StreamDataFeed
from bomber.framework.market.stream.health import (
    MarketHealthReason,
    MarketHealthSnapshot,
    MarketHealthState,
    StreamHealthConfig,
)
from bomber.framework.market.stream.aggregation import (
    QuoteMidBarFeed, ReceiveTimeTradeTickBarFeed, TradeTickBarFeed,
)

__all__ = [
    "StreamDataFeed",
    "MarketHealthReason",
    "MarketHealthSnapshot",
    "MarketHealthState",
    "StreamHealthConfig",
    "TradeTickBarFeed",
    "ReceiveTimeTradeTickBarFeed",
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
        from bomber.framework.market.stream.bn import BNWSConfig, BNWSStreamDataFeed

        return {
            "BNWSConfig": BNWSConfig,
            "BNWSStreamDataFeed": BNWSStreamDataFeed,
        }[name]

    if name in {"CtpMdConfig", "CtpLiveDataFeed", "CtpTickConverter"}:
        from bomber.framework.market.stream.ctp import CtpLiveDataFeed, CtpMdConfig, CtpTickConverter

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
        from bomber.framework.market.stream.dolphin import (
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
