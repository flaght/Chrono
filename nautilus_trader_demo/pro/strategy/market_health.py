"""把实时行情健康状态转换为策略级执行闸门。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from market.stream.health import (
    MarketHealthReason,
    MarketHealthSnapshot,
    MarketHealthState,
)
from strategy.contracts import TargetPortfolio, TargetUpdateMode


class StrategyMarketState(str, Enum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"


class MarketAccessMode(str, Enum):
    NORMAL = "NORMAL"
    REDUCE_ONLY = "REDUCE_ONLY"


@dataclass(frozen=True)
class StrategyMarketSnapshot:
    strategy_id: str
    state: StrategyMarketState
    access_mode: MarketAccessMode
    dependent_feeds: frozenset[str]
    unhealthy_feeds: Mapping[str, MarketHealthSnapshot]
    confirmation_required: bool
    affected_feeds: frozenset[str]
    version: int
    updated_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unhealthy_feeds",
            MappingProxyType(dict(self.unhealthy_feeds)),
        )


@dataclass(frozen=True)
class MarketGateDecision:
    access_mode: MarketAccessMode
    snapshot: StrategyMarketSnapshot


@dataclass(frozen=True)
class RecoveryConfirmation:
    strategy_id: str
    operator: str
    reason: str
    confirmed_ns: int
    gate_version: int
    feeds: frozenset[str]


class MarketHealthRejected(RuntimeError):
    def __init__(
        self,
        snapshot: StrategyMarketSnapshot,
        unsafe_targets: Mapping[str, tuple[Decimal, Decimal]],
    ) -> None:
        self.snapshot = snapshot
        self.unsafe_targets = MappingProxyType(dict(unsafe_targets))
        detail = ", ".join(
            f"{key}:{current}->{proposed}"
            for key, (current, proposed) in self.unsafe_targets.items()
        )
        super().__init__(
            f"策略{snapshot.strategy_id}行情状态为{snapshot.state.value}，"
            f"只允许降低风险；拒绝目标: {detail}",
        )


class MarketRecoveryError(RuntimeError):
    pass


@dataclass
class _FeedStatus:
    snapshot: MarketHealthSnapshot
    ever_ready: bool


@dataclass
class _StrategyStatus:
    feed_ids: frozenset[str]
    confirmation_required: bool = False
    affected_feeds: frozenset[str] = frozenset()
    degraded_since_ns: int | None = None


class MarketHealthGate:
    """按策略依赖关系隔离行情故障，并在异常期间只允许减风险目标。"""

    def __init__(self) -> None:
        self._feeds: dict[str, _FeedStatus] = {}
        self._strategies: dict[str, _StrategyStatus] = {}
        self._confirmations: list[RecoveryConfirmation] = []
        self._version = 0
        self._lock = threading.RLock()

    def register_feed(self, feed_id: str, snapshot: MarketHealthSnapshot) -> None:
        if not feed_id.strip():
            raise ValueError("feed_id不能为空")
        with self._lock:
            self._feeds[feed_id] = _FeedStatus(
                snapshot=snapshot,
                ever_ready=snapshot.state is MarketHealthState.READY,
            )
            self._version += 1

    def register_strategy(self, strategy_id: str, feed_ids: frozenset[str]) -> None:
        if not strategy_id.strip():
            raise ValueError("strategy_id不能为空")
        with self._lock:
            self._strategies[strategy_id] = _StrategyStatus(frozenset(feed_ids))
            self._version += 1

    def on_feed_health(self, feed_id: str, snapshot: MarketHealthSnapshot) -> None:
        with self._lock:
            previous = self._feeds.get(feed_id)
            was_ready = previous is not None and previous.snapshot.state is MarketHealthState.READY
            ever_ready = (
                snapshot.state is MarketHealthState.READY
                or (previous is not None and previous.ever_ready)
            )
            self._feeds[feed_id] = _FeedStatus(snapshot, ever_ready)
            # Runner.stop() 主动断开属于正常生命周期，不能留下需要人工解除的
            # 故障锁；真正的断流、超时、队列溢出和连接异常才进入恢复流程。
            should_latch = (
                was_ready
                and snapshot.state is not MarketHealthState.READY
                and snapshot.reason is not MarketHealthReason.EXPLICIT_DISCONNECT
            )
            if should_latch:
                for status in self._strategies.values():
                    if feed_id not in status.feed_ids:
                        continue
                    status.confirmation_required = True
                    status.affected_feeds = status.affected_feeds | {feed_id}
                    status.degraded_since_ns = max(
                        status.degraded_since_ns or 0,
                        snapshot.updated_ns,
                    )
            self._version += 1

    def snapshot(self, strategy_id: str) -> StrategyMarketSnapshot:
        with self._lock:
            status = self._require_strategy(strategy_id)
            unhealthy = {
                feed_id: self._feeds[feed_id].snapshot
                for feed_id in status.feed_ids
                if feed_id in self._feeds
                and self._feeds[feed_id].snapshot.state is not MarketHealthState.READY
            }
            if unhealthy:
                state = StrategyMarketState.DEGRADED
            elif status.confirmation_required:
                state = StrategyMarketState.AWAITING_CONFIRMATION
            else:
                state = StrategyMarketState.READY
            access = (
                MarketAccessMode.NORMAL
                if state is StrategyMarketState.READY
                else MarketAccessMode.REDUCE_ONLY
            )
            return StrategyMarketSnapshot(
                strategy_id=strategy_id,
                state=state,
                access_mode=access,
                dependent_feeds=status.feed_ids,
                unhealthy_feeds=unhealthy,
                confirmation_required=status.confirmation_required,
                affected_feeds=status.affected_feeds,
                version=self._version,
                updated_ns=time.time_ns(),
            )

    def check_target(
        self,
        intent: TargetPortfolio,
        previous: TargetPortfolio | None,
    ) -> MarketGateDecision:
        snapshot = self.snapshot(intent.strategy_id)
        if snapshot.access_mode is MarketAccessMode.NORMAL:
            return MarketGateDecision(MarketAccessMode.NORMAL, snapshot)

        current = dict(previous.targets) if previous is not None else {}
        if intent.update_mode is TargetUpdateMode.PATCH:
            proposed = dict(current)
            proposed.update(intent.targets)
        else:
            proposed = dict(intent.targets)
        unsafe: dict[str, tuple[Decimal, Decimal]] = {}
        for target_key in current.keys() | proposed.keys():
            before = current.get(target_key, Decimal(0))
            after = proposed.get(target_key, Decimal(0))
            if not _does_not_increase_risk(before, after):
                unsafe[target_key] = (before, after)
        if unsafe:
            raise MarketHealthRejected(snapshot, unsafe)
        return MarketGateDecision(MarketAccessMode.REDUCE_ONLY, snapshot)

    def confirm_recovery(
        self,
        strategy_id: str,
        *,
        operator: str,
        reason: str,
        confirmed_ns: int | None = None,
    ) -> RecoveryConfirmation:
        if not operator.strip() or not reason.strip():
            raise ValueError("人工恢复确认必须记录operator和reason")
        with self._lock:
            status = self._require_strategy(strategy_id)
            if not status.confirmation_required:
                raise MarketRecoveryError("策略当前不需要人工恢复确认")
            snapshot = self.snapshot(strategy_id)
            if snapshot.unhealthy_feeds:
                raise MarketRecoveryError(
                    "依赖行情尚未全部READY: "
                    + ", ".join(sorted(snapshot.unhealthy_feeds)),
                )
            degraded_since = status.degraded_since_ns or 0
            stale_recoveries = []
            for feed_id in status.affected_feeds:
                feed = self._feeds.get(feed_id)
                received_ns = None if feed is None else feed.snapshot.last_event_received_ns
                if received_ns is None or received_ns <= degraded_since:
                    stale_recoveries.append(feed_id)
            if stale_recoveries:
                raise MarketRecoveryError(
                    "行情虽标记READY，但故障后尚无新鲜事件: "
                    + ", ".join(sorted(stale_recoveries)),
                )
            status.confirmation_required = False
            feeds = status.affected_feeds
            status.affected_feeds = frozenset()
            status.degraded_since_ns = None
            self._version += 1
            confirmation = RecoveryConfirmation(
                strategy_id=strategy_id,
                operator=operator,
                reason=reason,
                confirmed_ns=time.time_ns() if confirmed_ns is None else confirmed_ns,
                gate_version=self._version,
                feeds=feeds,
            )
            self._confirmations.append(confirmation)
            return confirmation

    @property
    def confirmations(self) -> tuple[RecoveryConfirmation, ...]:
        with self._lock:
            return tuple(self._confirmations)

    def _require_strategy(self, strategy_id: str) -> _StrategyStatus:
        try:
            return self._strategies[strategy_id]
        except KeyError as exc:
            raise ValueError(f"未知strategy_id: {strategy_id}") from exc


def _does_not_increase_risk(current: Decimal, proposed: Decimal) -> bool:
    if proposed == current:
        return True
    if current == 0:
        return proposed == 0
    return abs(proposed) < abs(current) and current * proposed >= 0
