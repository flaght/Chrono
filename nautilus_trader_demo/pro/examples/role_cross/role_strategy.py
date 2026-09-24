"""第四类M3a：真实合约Bar触发，DataHub研究价产生单一逻辑目标。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from datahub import ResearchDataUnavailable, RoleSnapshot
from market.basic.base import Bar
from trader.template import StrategyTemplate

from .role_signal import RoleCrossEvent, RoleCrossSignal


class RoleSnapshotPort(Protocol):
    def snapshot(self, as_of_ns: int, *, max_source_age_ns: int | None = None) -> RoleSnapshot: ...


@dataclass(frozen=True)
class RoleCrossConfig:
    target_key: str = "rb_main"
    quantity: Decimal = Decimal(1)
    venue: str = "SHFE"
    max_source_age_ns: int | None = None

    def __post_init__(self) -> None:
        quantity = Decimal(str(self.quantity))
        if not self.target_key.strip() or not self.venue.strip():
            raise ValueError("逻辑目标键和交易所不能为空")
        if not quantity.is_finite() or quantity <= 0:
            raise ValueError("目标手数必须为正且有限")
        if self.max_source_age_ns is not None and self.max_source_age_ns < 0:
            raise ValueError("最大源数据年龄不能为负")
        object.__setattr__(self, "quantity", quantity)


class RoleCrossTargetStrategy(StrategyTemplate):
    """等待四张角色真实合约的同时间戳Bar到齐，然后只提交`rb_main`目标。

    数据文件、主力解析、复权因子、撤单和平旧仓均不属于策略职责。当前
    M3阶段刻意要求完整同分钟帧；低流动性缺Bar的受控前填另行验收。
    """

    def __init__(self, strategy_id: str, data_hub: RoleSnapshotPort,
                 config: RoleCrossConfig | None = None) -> None:
        super().__init__(strategy_id)
        self.data_hub = data_hub
        self.config = config or RoleCrossConfig()
        self.signal = RoleCrossSignal()
        self.complete_frames = 0
        self.unavailable_events = 0
        self.signal_events: list[RoleCrossEvent] = []
        self._pending_ns = -1
        self._arrived: set[str] = set()

    def on_bar(self, data_key: str, bar: Bar) -> None:
        del data_key  # 数据键由Runner装配；策略只关心标准Bar的真实合约。
        timestamp = int(bar.ts_event)
        if timestamp <= self.signal.last_processed_ns or timestamp < self._pending_ns:
            return
        if timestamp != self._pending_ns:
            self._pending_ns = timestamp
            self._arrived.clear()
        instrument_id = bar.bar_type.instrument_id
        if str(instrument_id.venue).upper() != self.config.venue.upper():
            return
        self._arrived.add(str(instrument_id.symbol).lower())
        try:
            snapshot = self.data_hub.snapshot(
                timestamp, max_source_age_ns=self.config.max_source_age_ns,
            )
        except ResearchDataUnavailable:
            self.unavailable_events += 1
            return
        expected = {contract.lower() for contract in snapshot.contracts.values()}
        if not expected.issubset(self._arrived):
            return
        # 文件预加载可能使同时间戳价格在首个Bar回调前就能查到；仍须等
        # 四个实际Bar事件都到齐，不允许对尚未送达的Bar提前决策。
        if any(snapshot.prices[role].source_ns != timestamp for role in snapshot.prices):
            return
        self.complete_frames += 1
        event = self.signal.update(snapshot)
        if event is None:
            return
        self.signal_events.append(event)
        self.set_target(
            self.config.target_key,
            self.config.quantity * event.direction,
            event.ts_event,
            metadata={
                "signal_direction": event.direction,
                "signal_spread": str(event.spread),
                "research_main": event.main_instrument,
            },
        )
