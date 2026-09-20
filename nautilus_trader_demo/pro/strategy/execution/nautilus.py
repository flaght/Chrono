"""Bomber/Nautilus原生订单执行适配器。"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bomber.model.enums import OrderSide
from bomber.model.identifiers import ClientOrderId, InstrumentId
from bomber.model.instruments import Instrument
from bomber.model.objects import Quantity

from strategy.contracts import TargetPortfolio


class NautilusExecutionAdapter:
    """依托原生Strategy宿主，把单标的目标仓位协调为差额市价单。"""

    def __init__(
        self,
        host: Any,
        *,
        strategy_id: str,
        target_key: str,
        instrument_id: InstrumentId,
        submit_orders: bool,
    ) -> None:
        self.host = host
        self.strategy_id = strategy_id
        self.target_key = target_key
        self.instrument_id = instrument_id
        self.submit_orders = submit_orders
        self.instrument: Instrument | None = None
        self.desired_target: Decimal | None = None
        self.pending_order_id: ClientOrderId | None = None
        self.target_events: list[dict[str, str | int]] = []
        self.order_events: list[dict[str, str | int]] = []
        self._last_revision = 0

    def start(self, instrument: Instrument) -> None:
        self.instrument = instrument

    def position(self, target_key: str) -> Decimal:
        if target_key != self.target_key:
            raise ValueError(f"未知target_key: {target_key}")
        return Decimal(str(self.host.portfolio.net_position(self.instrument_id)))

    def submit_target(self, intent: TargetPortfolio) -> None:
        if intent.strategy_id != self.strategy_id:
            raise ValueError(f"目标策略不匹配: {intent.strategy_id}")
        if intent.revision <= self._last_revision:
            raise ValueError(
                f"目标revision必须递增: {intent.revision} <= {self._last_revision}",
            )
        unknown = set(intent.targets) - {self.target_key}
        if unknown:
            raise ValueError(f"执行适配器没有这些目标路由: {sorted(unknown)}")
        self._last_revision = intent.revision
        self.desired_target = intent.targets.get(self.target_key, Decimal(0))
        self.target_events.append(
            {
                "timestamp_ns": intent.ts_event,
                "revision": intent.revision,
                "target": str(self.desired_target),
                "signal": str(intent.metadata.get("signal", "")),
            },
        )

    def reconcile(self, timestamp_ns: int) -> None:
        if not self.submit_orders:
            return
        if self.desired_target is None or self.pending_order_id is not None:
            return
        if self.instrument is None:
            raise RuntimeError("执行适配器尚未启动")
        current = Decimal(str(self.host.portfolio.net_position(self.instrument_id)))
        delta = self.desired_target - current
        if delta == 0:
            return
        reducing = current != 0 and (
            self.desired_target == 0
            or (current * self.desired_target > 0 and abs(self.desired_target) < abs(current))
        )
        quantity: Quantity = self.instrument.make_qty(abs(delta))
        order = self.host.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
            quantity=quantity,
            reduce_only=reducing,
        )
        self.pending_order_id = order.client_order_id
        self.order_events.append(
            {
                "timestamp_ns": timestamp_ns,
                "event": "submitted",
                "current": str(current),
                "target": str(self.desired_target),
                "delta": str(delta),
                "client_order_id": str(order.client_order_id),
            },
        )
        self.host.submit_order(order)

    def order_filled(self, event: Any) -> None:
        if event.client_order_id != self.pending_order_id:
            return
        order = self.host.cache.order(event.client_order_id)
        self.order_events.append(
            {
                "timestamp_ns": event.ts_event,
                "event": "filled",
                "quantity": str(event.last_qty),
                "client_order_id": str(event.client_order_id),
            },
        )
        if order is not None and order.is_closed:
            self.pending_order_id = None
            self.reconcile(event.ts_event)

    def order_finished(self, client_order_id: ClientOrderId, timestamp_ns: int, name: str) -> None:
        if client_order_id != self.pending_order_id:
            return
        self.order_events.append(
            {
                "timestamp_ns": timestamp_ns,
                "event": name,
                "client_order_id": str(client_order_id),
            },
        )
        self.pending_order_id = None
