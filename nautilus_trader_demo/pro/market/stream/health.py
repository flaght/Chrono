"""实时行情源的通用健康状态与异常检测。"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


logger = logging.getLogger("MarketStreamHealth")


class MarketHealthState(str, Enum):
    """策略运行期可依赖的三态行情健康模型。"""

    READY = "ready"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


class MarketHealthReason(str, Enum):
    """最近一次健康状态变化的原因。"""

    INITIAL = "initial"
    WAITING_FOR_FIRST_EVENT = "waiting_for_first_event"
    HEALTHY = "healthy"
    STALE = "stale"
    STREAM_INTERRUPTED = "stream_interrupted"
    QUEUE_OVERFLOW = "queue_overflow"
    TIMESTAMP_ROLLBACK = "timestamp_rollback"
    DISPATCH_ERROR = "dispatch_error"
    CONNECTION_ERROR = "connection_error"
    EXPLICIT_DISCONNECT = "explicit_disconnect"


@dataclass(frozen=True)
class StreamHealthConfig:
    """实时流健康检测参数。

    ``startup_grace_seconds`` 是连接后等待首条行情的宽限期；
    ``stale_after_seconds`` 是收到过行情后的最大静默时间。时间戳回退按
    “事件类型 + 标的 + Bar类型”分别检查，避免多个合约交错到达时误报。
    """

    startup_grace_seconds: float = 30.0
    stale_after_seconds: float = 30.0
    timestamp_rollback_tolerance_ns: int = 0
    drop_timestamp_rollback: bool = True

    def __post_init__(self) -> None:
        if self.startup_grace_seconds <= 0:
            raise ValueError("startup_grace_seconds must be positive")
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        if self.timestamp_rollback_tolerance_ns < 0:
            raise ValueError("timestamp_rollback_tolerance_ns must be non-negative")


@dataclass(frozen=True)
class MarketHealthSnapshot:
    """可安全跨线程读取和保存的行情健康快照。"""

    source_id: str
    state: MarketHealthState
    reason: MarketHealthReason
    detail: str
    version: int
    updated_ns: int
    connected_at_ns: int | None
    last_event_received_ns: int | None
    last_event_ts: int | None
    queue_overflows: int
    timestamp_rollbacks: int
    dispatch_errors: int
    stream_interruptions: int


HealthHandler = Callable[[MarketHealthSnapshot], None]


class StreamHealthMonitor:
    """由所有在线Feed复用的线程安全健康状态机。"""

    def __init__(
        self,
        source_id: str,
        config: StreamHealthConfig | None = None,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.source_id = source_id
        self.config = config or StreamHealthConfig()
        self._monotonic_ns = monotonic_ns
        self._wall_time_ns = wall_time_ns
        self._lock = threading.RLock()
        self._handlers: list[HealthHandler] = []
        self._connected = False
        self._connected_monotonic_ns: int | None = None
        self._connected_wall_ns: int | None = None
        self._last_event_monotonic_ns: int | None = None
        self._last_event_wall_ns: int | None = None
        self._last_event_ts_by_key: dict[tuple[str, str, str], int] = {}
        self._sticky_reason: MarketHealthReason | None = None
        self._version = 0
        self._queue_overflows = 0
        self._timestamp_rollbacks = 0
        self._dispatch_errors = 0
        self._stream_interruptions = 0
        self._snapshot = self._make_snapshot(
            MarketHealthState.DISCONNECTED,
            MarketHealthReason.INITIAL,
            "feed尚未连接",
        )

    @property
    def snapshot(self) -> MarketHealthSnapshot:
        with self._lock:
            return self._snapshot

    def register_handler(self, handler: HealthHandler) -> None:
        with self._lock:
            self._handlers.append(handler)

    def on_connected(self) -> None:
        with self._lock:
            now = self._monotonic_ns()
            self._connected = True
            self._connected_monotonic_ns = now
            self._connected_wall_ns = self._wall_time_ns()
            self._last_event_monotonic_ns = None
            self._last_event_wall_ns = None
            self._last_event_ts_by_key.clear()
            self._sticky_reason = None
            notification = self._transition(
                MarketHealthState.DEGRADED,
                MarketHealthReason.WAITING_FOR_FIRST_EVENT,
                "网络已连接，等待首条行情",
                force=True,
            )
        self._notify(notification)

    def on_disconnected(
        self,
        reason: MarketHealthReason = MarketHealthReason.EXPLICIT_DISCONNECT,
        detail: str = "feed已断开",
    ) -> None:
        with self._lock:
            self._connected = False
            self._connected_monotonic_ns = None
            self._connected_wall_ns = None
            self._last_event_monotonic_ns = None
            self._last_event_wall_ns = None
            self._last_event_ts_by_key.clear()
            self._sticky_reason = None
            notification = self._transition(
                MarketHealthState.DISCONNECTED,
                reason,
                detail,
                force=True,
            )
        self._notify(notification)

    def on_event(self, event: Any) -> bool:
        """登记事件并返回是否允许该事件进入分发队列。"""
        event_ts = _event_timestamp(event)
        key = _event_key(event)
        notification: MarketHealthSnapshot | None = None
        accepted = True
        with self._lock:
            if event_ts is not None:
                previous = self._last_event_ts_by_key.get(key)
                tolerance = self.config.timestamp_rollback_tolerance_ns
                if previous is not None and event_ts + tolerance < previous:
                    self._timestamp_rollbacks += 1
                    self._sticky_reason = MarketHealthReason.TIMESTAMP_ROLLBACK
                    notification = self._transition(
                        MarketHealthState.DEGRADED,
                        MarketHealthReason.TIMESTAMP_ROLLBACK,
                        f"{key[1]} {key[0]}时间戳从{previous}回退到{event_ts}",
                        force=True,
                    )
                    accepted = not self.config.drop_timestamp_rollback
                else:
                    self._last_event_ts_by_key[key] = max(previous or event_ts, event_ts)

            if accepted:
                self._last_event_monotonic_ns = self._monotonic_ns()
                self._last_event_wall_ns = self._wall_time_ns()
                if self._connected and self._sticky_reason is None:
                    notification = self._transition(
                        MarketHealthState.READY,
                        MarketHealthReason.HEALTHY,
                        "行情持续到达",
                    )
        self._notify(notification)
        return accepted

    def check_timeout(self) -> MarketHealthSnapshot:
        with self._lock:
            if not self._connected or self._sticky_reason is not None:
                return self._snapshot
            now = self._monotonic_ns()
            if self._last_event_monotonic_ns is None:
                assert self._connected_monotonic_ns is not None
                elapsed = now - self._connected_monotonic_ns
                threshold = int(self.config.startup_grace_seconds * 1_000_000_000)
                detail = "连接后未收到首条行情"
            else:
                elapsed = now - self._last_event_monotonic_ns
                threshold = int(self.config.stale_after_seconds * 1_000_000_000)
                detail = "行情流超过允许时间未更新"
            notification = None
            if elapsed >= threshold:
                notification = self._transition(
                    MarketHealthState.DEGRADED,
                    MarketHealthReason.STALE,
                    detail,
                )
            snapshot = self._snapshot
        self._notify(notification)
        return snapshot

    def on_queue_overflow(self) -> None:
        self._record_sticky(
            MarketHealthReason.QUEUE_OVERFLOW,
            "消费队列已满，至少一条行情被丢弃",
            "_queue_overflows",
        )

    def on_dispatch_error(self, error: BaseException) -> None:
        self._record_sticky(
            MarketHealthReason.DISPATCH_ERROR,
            f"行情事件分派异常: {error}",
            "_dispatch_errors",
        )

    def on_stream_interrupted(self, detail: str) -> None:
        with self._lock:
            if not self._connected:
                return
            self._stream_interruptions += 1
            notification = self._transition(
                MarketHealthState.DEGRADED,
                MarketHealthReason.STREAM_INTERRUPTED,
                detail or "底层行情流中断",
                force=True,
            )
        self._notify(notification)

    def acknowledge_degradation(self) -> MarketHealthSnapshot:
        """运维完成补洞/核查后，清除需要人工确认的粘滞故障。"""
        with self._lock:
            self._sticky_reason = None
            if not self._connected:
                return self._snapshot
            now = self._monotonic_ns()
            if self._last_event_monotonic_ns is None:
                reason = MarketHealthReason.WAITING_FOR_FIRST_EVENT
                state = MarketHealthState.DEGRADED
                detail = "已确认故障，仍在等待首条行情"
            else:
                elapsed = now - self._last_event_monotonic_ns
                threshold = int(self.config.stale_after_seconds * 1_000_000_000)
                if elapsed >= threshold:
                    reason = MarketHealthReason.STALE
                    state = MarketHealthState.DEGRADED
                    detail = "已确认故障，但行情仍已超时"
                else:
                    reason = MarketHealthReason.HEALTHY
                    state = MarketHealthState.READY
                    detail = "故障已核查，行情仍在时效范围内"
            notification = self._transition(state, reason, detail, force=True)
            snapshot = self._snapshot
        self._notify(notification)
        return snapshot

    def _record_sticky(
        self,
        reason: MarketHealthReason,
        detail: str,
        counter_name: str,
    ) -> None:
        with self._lock:
            setattr(self, counter_name, getattr(self, counter_name) + 1)
            self._sticky_reason = reason
            notification = self._transition(
                MarketHealthState.DEGRADED,
                reason,
                detail,
                force=True,
            )
        self._notify(notification)

    def _transition(
        self,
        state: MarketHealthState,
        reason: MarketHealthReason,
        detail: str,
        *,
        force: bool = False,
    ) -> MarketHealthSnapshot | None:
        current = self._snapshot
        if not force and current.state is state and current.reason is reason:
            return None
        self._version += 1
        self._snapshot = self._make_snapshot(state, reason, detail)
        return self._snapshot

    def _make_snapshot(
        self,
        state: MarketHealthState,
        reason: MarketHealthReason,
        detail: str,
    ) -> MarketHealthSnapshot:
        last_ts = max(self._last_event_ts_by_key.values(), default=None)
        return MarketHealthSnapshot(
            source_id=self.source_id,
            state=state,
            reason=reason,
            detail=detail,
            version=self._version,
            updated_ns=self._wall_time_ns(),
            connected_at_ns=self._connected_wall_ns,
            last_event_received_ns=self._last_event_wall_ns,
            last_event_ts=last_ts,
            queue_overflows=self._queue_overflows,
            timestamp_rollbacks=self._timestamp_rollbacks,
            dispatch_errors=self._dispatch_errors,
            stream_interruptions=self._stream_interruptions,
        )

    def _notify(self, snapshot: MarketHealthSnapshot | None) -> None:
        if snapshot is None:
            return
        with self._lock:
            handlers = tuple(self._handlers)
        for handler in handlers:
            try:
                handler(snapshot)
            except Exception:
                logger.exception("行情健康状态处理器异常: source=%s", self.source_id)


def _event_timestamp(event: Any) -> int | None:
    value = getattr(event, "ts_event", None)
    if value is None:
        return None
    return int(value)


def _event_key(event: Any) -> tuple[str, str, str]:
    instrument = str(getattr(event, "instrument_id", ""))
    bar_type = getattr(event, "bar_type", None)
    if bar_type is None:
        nested_bar = getattr(event, "bar", None)
        bar_type = getattr(nested_bar, "bar_type", "")
    bar_type = str(bar_type)
    return type(event).__name__, instrument, bar_type
