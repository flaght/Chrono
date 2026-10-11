"""通用时效及流健康检查；CTP深度与交易日规则属于channels/ctp。"""

import math
import time

from .contracts import ReadinessIssue, ReadinessResult
from .profiles import MarketTimeProfile


def validate_instrument_max_age(seconds):
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError("instrument-max-age-seconds须为(0,120]内有限数")
    value = int(seconds * 1_000_000_000)
    if value <= 0:
        raise ValueError("时效须至少为1纳秒")
    return value


def timestamp_issue(source, label, timestamp, now_ns, max_age_ns, *, allow_historical=False):
    if timestamp is None or timestamp <= 0:
        return ReadinessIssue(source, f"{label}_MISSING", "行情时间缺失")
    age = now_ns - timestamp
    context = {"age_seconds": age / 1e9, "max_age_seconds": max_age_ns / 1e9}
    if age < 0:
        return ReadinessIssue(source, f"{label}_FUTURE", "行情时间来自未来", context)
    if age > max_age_ns and not allow_historical:
        return ReadinessIssue(source, f"{label}_STALE", "行情超过允许时效", context)
    return None


class StreamHealthCheck:
    """适用于提供标准health_snapshot的流；不包含执行端或交易日判断。"""
    def __init__(self, upstream, identity, *, profile=None, max_age_seconds=10,
                 clock_ns=None, monotonic_ns=None):
        if identity.domain != "market":
            raise ValueError("流健康检查只接受行情来源")
        self.upstream, self.identity = upstream, identity
        self.profile = profile or MarketTimeProfile()
        self.max_age_ns = validate_instrument_max_age(max_age_seconds)
        self.clock_ns = clock_ns or (lambda: time.time_ns())
        self.monotonic_ns = monotonic_ns or (lambda: time.monotonic_ns())

    def check(self):
        health = self.upstream.health_snapshot
        issues = []
        if getattr(health.state, "value", health.state) != "ready":
            issues.append(ReadinessIssue(self.identity, "STREAM_UNHEALTHY",
                str(getattr(health, "reason", health.state)),
                {"detail": getattr(health, "detail", "")}))
        now = self.clock_ns()
        if hasattr(self.upstream, "latest_receive_monotonic_ns"):
            received = self.upstream.latest_receive_monotonic_ns
            receive_now = self.monotonic_ns()
        else:
            received = getattr(health, "last_event_received_ns", None)
            receive_now = now
        for issue in (
            timestamp_issue(self.identity, "RECEIVE", received, receive_now, self.max_age_ns),
            timestamp_issue(self.identity, "EVENT", getattr(health, "last_event_ts", None),
                            now, self.max_age_ns,
                            allow_historical=self.profile.allow_historical_events),
        ):
            if issue is not None:
                issues.append(issue)
        return ReadinessResult((self.identity,), tuple(issues))

