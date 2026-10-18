"""真实合约分钟行情触发，参考资料研究价产生主力逻辑目标。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from bomber.framework.datahub import ResearchDataUnavailable, RoleSnapshot
from bomber.framework.market.basic.base import Bar
from bomber.framework.trader.template import StrategyTemplate

from role_signal import RoleCrossEvent, RoleCrossSignal


class RoleSnapshotPort(Protocol):
    """策略依赖的参考快照接口，不绑定具体文件或数据存储实现。"""
    def snapshot(self, as_of_ns: int, *, max_source_age_ns: int | None = None) -> RoleSnapshot: ...


@dataclass(frozen=True)
class RoleCrossConfig:
    """配置主力逻辑目标、交易所、整数手数和研究价格时效。"""
    target_key: str
    venue: str
    quantity: Decimal = Decimal(1)
    max_source_age_ns: int | None = None

    def __post_init__(self) -> None:
        quantity = Decimal(str(self.quantity))
        if not self.target_key.strip() or not self.venue.strip():
            raise ValueError("逻辑目标键和交易所不能为空")
        if not quantity.is_finite() or quantity <= 0 or quantity != quantity.to_integral_value():
            raise ValueError("期货目标手数必须为正整数且有限")
        if self.max_source_age_ns is not None and self.max_source_age_ns < 0:
            raise ValueError("最大源数据年龄不能为负")
        object.__setattr__(self, "quantity", quantity)


class RoleCrossTargetStrategy(StrategyTemplate):
    """等待主力、次主力、远期真实合约的同时间戳行情到齐，再提交主力逻辑目标。

    数据文件与因子由公共输入层及参考资料服务准备，主力换约由执行路由处理。
    要求完整同分钟帧，不对低流动性缺失行情进行前向填充。
    """

    def __init__(self, strategy_id: str, data_hub: RoleSnapshotPort,
                 config: RoleCrossConfig) -> None:
        super().__init__(strategy_id)
        self.data_hub = data_hub
        self.config = config
        self.signal = RoleCrossSignal()
        self.complete_frames = 0
        self.unavailable_events = 0
        self.signal_events: list[RoleCrossEvent] = []
        self._pending_ns = -1
        self._arrived: set[str] = set()

    def on_bar(self, data_key: str, bar: Bar) -> None:
        if data_key != str(bar.bar_type.instrument_id):
            raise ValueError(f"行情合约与数据键不匹配: {data_key}")
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
        # 文件预加载可能使研究价提前可查询；必须等三个真实行情事件到齐。
        if any(snapshot.prices[role].source_ns != timestamp for role in snapshot.prices):
            return
        self.complete_frames += 1
        event = self.signal.update(snapshot)
        if event is None:
            return
        self.signal_events.append(event)
        # 仅提交一个主力逻辑目标；解析真实主力、处理旧仓由动态执行路由负责。
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
