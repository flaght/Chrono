"""在线环境时间规则及逐合约深度门控；无引擎、网络或策略依赖。"""

from dataclasses import dataclass
from datetime import datetime
import math
import time
from urllib.parse import urlsplit


@dataclass(frozen=True)
class LiveTimePolicy:
    environment: str = "realtime"
    replay_md_trading_day: str | None = None

    def __post_init__(self):
        if self.environment not in {"realtime", "replay"}:
            raise ValueError("未知SimNow环境")
        if self.environment == "replay":
            value = self.replay_md_trading_day or ""
            if (len(value) != 8 or not value.isdigit() or
                    datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") != value):
                raise ValueError("replay须提供有效的--replay-md-trading-day YYYYMMDD")
        elif self.replay_md_trading_day:
            raise ValueError("--replay-md-trading-day仅用于replay模式")

    @property
    def bar_time_basis(self):
        return "receive" if self.environment == "replay" else "event"

    def expected_md_day(self, trading_day):
        return self.replay_md_trading_day or trading_day

    def tick_timestamp(self, tick):
        return tick.ts_init if self.environment == "replay" else tick.ts_event


def validate_replay_environment(md_front, td_front, production_mode):
    for front, port in ((md_front, 40011), (td_front, 40001)):
        parsed = urlsplit(front)
        if (parsed.scheme != "tcp" or parsed.hostname != "182.254.243.31" or parsed.port != port
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("replay模式须使用同组SimNow MD40011／TD40001前置")
    if not production_mode:
        raise ValueError("本次第二套看穿式前置须CTP_PRODUCTION_MODE=true")


def add_environment_arguments(parser):
    parser.add_argument("--simnow-environment", choices=("realtime", "replay"), default="realtime",
                        help="realtime检查事件时效；replay按接收时间聚合第二套SimNow行情")
    parser.add_argument("--replay-md-trading-day", help="replay必填：原始行情TradingDay，YYYYMMDD")
    parser.add_argument("--instrument-max-age-seconds", type=float, default=10,
                        help="逐合约时效(0,120]秒；realtime检查事件/接收，replay检查接收")


def validate_environment_arguments(args):
    policy = LiveTimePolicy(args.simnow_environment, args.replay_md_trading_day)
    validate_instrument_max_age(args.instrument_max_age_seconds)
    return policy


def validate_instrument_max_age(seconds):
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError("instrument-max-age-seconds须为(0,120]内有限数")
    return int(seconds * 1_000_000_000)


class InstrumentHealthGate:
    """原始深度必须覆盖每个固定合约；回放只豁免历史事件年龄上限。

    fallback仅供没有深度接口的手动Feed使用。真实CTP的空深度字典仍拒绝。
    """
    def __init__(self, upstream, instruments, expected_day, *, policy=None,
                 max_age_seconds=10, clock_ns=None, monotonic_ns=None,
                 fallback_events=None, fallback_receives=None):
        self.upstream = upstream
        self.instruments = frozenset(instruments)
        self.policy = policy or LiveTimePolicy()
        self.expected_day = self.policy.expected_md_day(expected_day)
        self.max_age_ns = validate_instrument_max_age(max_age_seconds)
        self.clock_ns = clock_ns or (lambda: time.time_ns())
        self.monotonic_ns = monotonic_ns or (lambda: time.monotonic_ns())
        self.fallback_events = fallback_events
        self.fallback_receives = fallback_receives
        self.depth_records = {}

    def stale_instruments(self):
        now, received = self.clock_ns(), self.monotonic_ns()
        observations = getattr(self.upstream, "depth_observations", None)
        if observations is None and self.fallback_events is None:
            return {}  # 非CTP Feed继续由自身健康接口门控。
        if observations:
            self.depth_records = dict(observations)
        failures = {}
        for instrument in self.instruments:
            observation = observations.get(instrument) if observations is not None else None
            event = (observation.ts_event if observation else
                     self.fallback_events.get(instrument) if observations is None else None)
            receive = (observation.received_monotonic_ns if observation else
                       (self.fallback_receives or {}).get(instrument) if observations is None else None)
            day_ok = observations is None or (observation and observation.trading_day == self.expected_day)
            regressed = bool(observation and observation.timestamp_regressed)
            event_age = None if event is None else now - event
            receive_age = None if receive is None else received - receive
            if (not day_ok or regressed or event_age is None or receive_age is None or
                    event_age < 0 or
                    (self.policy.environment == "realtime" and event_age > self.max_age_ns) or
                    not 0 <= receive_age <= self.max_age_ns):
                failures[str(instrument)] = {
                    "event_age_seconds": None if event_age is None else event_age / 1e9,
                    "receive_age_seconds": None if receive_age is None else receive_age / 1e9,
                    "trading_day": observation.trading_day if observation else None,
                    "max_age_seconds": self.max_age_ns / 1e9, "timestamp_regressed": regressed}
        return failures
