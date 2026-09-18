from typing import Any

from market.stream.base import StreamDataFeed

__all__ = [
    "StreamDataFeed",
    "BNWSConfig",
    "BNWSStreamDataFeed",
    "CtpMdConfig",
    "CtpLiveDataFeed",
    "CtpTickConverter",
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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")