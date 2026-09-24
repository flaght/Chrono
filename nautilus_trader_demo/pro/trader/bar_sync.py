"""多标的标准Bar的同时间戳同步器。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Mapping

from market.basic.base import Bar, InstrumentId


@dataclass(frozen=True)
class SynchronizedBarFrame:
    """所有指定标的在同一事件时间到齐的一帧Bar。"""

    ts_event: int
    bars: Mapping[InstrumentId, Bar]


class BarSynchronizer:
    """只输出完整帧；缺标的、重复或已经错过的旧帧不触发决策。

    不做前向填充。历史回放的各路事件应按时间顺序输入；若单路时间回退，
    明确报错而不是把旧数据混入新帧。
    """

    def __init__(self, instruments: tuple[InstrumentId, ...]) -> None:
        if len(instruments) < 2 or len(set(instruments)) != len(instruments):
            raise ValueError("Bar同步至少需要两个不重复的标的")
        self.instruments = tuple(instruments)
        self._expected = frozenset(instruments)
        self._latest: dict[InstrumentId, Bar] = {}
        self._last_emitted_ns = -1
        self._lock = RLock()

    def push(self, bar: Bar) -> SynchronizedBarFrame | None:
        instrument_id = bar.bar_type.instrument_id
        if instrument_id not in self._expected:
            raise ValueError(f"同步器未配置标的: {instrument_id}")
        with self._lock:
            previous = self._latest.get(instrument_id)
            if previous is not None:
                if bar.ts_event < previous.ts_event:
                    raise ValueError(f"Bar时间回退: {instrument_id}")
                if bar.ts_event == previous.ts_event:
                    return None
            self._latest[instrument_id] = bar
            if len(self._latest) != len(self.instruments):
                return None
            timestamp = bar.ts_event
            if timestamp <= self._last_emitted_ns or any(
                self._latest[item].ts_event != timestamp for item in self.instruments
            ):
                return None
            self._last_emitted_ns = timestamp
            return SynchronizedBarFrame(
                timestamp,
                MappingProxyType({item: self._latest[item] for item in self.instruments}),
            )
