"""与渠道无关的时间语义和时段组合，不读取环境变量或柜台身份。"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class MarketTimeProfile:
    environment: str = "realtime"

    def __post_init__(self):
        if self.environment not in {"realtime", "replay"}:
            raise ValueError("未知在线行情时间模式")

    @property
    def bar_time_basis(self):
        return "receive" if self.environment == "replay" else "event"

    @property
    def allow_historical_events(self):
        return self.environment == "replay"

    def tick_timestamp(self, tick):
        return tick.ts_init if self.allow_historical_events else tick.ts_event


@dataclass(frozen=True)
class LiveProfile:
    time: MarketTimeProfile = field(default_factory=MarketTimeProfile)
    session_window: Callable[[int], bool] | None = None

    def is_open(self, now_ns):
        return self.session_window is None or bool(self.session_window(now_ns))


def build_minute_feed(source_id, upstream, policy, *, after_tick=None,
                      max_age_seconds=10, clock_ns=None):
    """独立行情积木工厂；不选择渠道、不创建任何执行端。"""
    from .health import validate_instrument_max_age
    from bomber.framework.market.stream import ReceiveTimeTradeTickBarFeed, TradeTickBarFeed
    kwargs = {"after_tick": after_tick}
    if policy.bar_time_basis == "receive":
        kwargs.update(max_receive_age_ns=validate_instrument_max_age(max_age_seconds), clock_ns=clock_ns)
        return ReceiveTimeTradeTickBarFeed(source_id, upstream, **kwargs)
    return TradeTickBarFeed(source_id, upstream, **kwargs)
