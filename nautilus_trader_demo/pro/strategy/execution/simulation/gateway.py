"""中立OrderIntent与Nautilus原生Strategy订单API之间的内部网关。"""

from __future__ import annotations

from decimal import Decimal
from typing import Callable

from bomber.model.enums import OrderSide as NativeOrderSide
from bomber.model.events import (
    OrderAccepted,
    OrderCanceled,
    OrderDenied,
    OrderExpired,
    OrderFilled,
    OrderRejected,
)
from bomber.model.identifiers import ClientOrderId
from bomber.trading.config import StrategyConfig
from bomber.trading.strategy import Strategy

from strategy.execution.contracts import (
    ExecutionReport,
    ExecutionReportType,
    OrderIntent,
    OrderSide,
    OrderType,
    PositionEffect,
)


class NautilusOrderGateway(Strategy):
    """Backend内部唯一可以调用Nautilus原生下单API的Strategy宿主。"""

    def __init__(
        self,
        backend_id: str,
        report_sink: Callable[[ExecutionReport], None],
    ) -> None:
        super().__init__(StrategyConfig())
        self._backend_id = backend_id
        self._report_sink = report_sink
        self._ready = False
        self._queued: list[OrderIntent] = []
        self._intents: dict[ClientOrderId, OrderIntent] = {}

    def enqueue(self, intent: OrderIntent) -> None:
        if self._ready:
            self._submit(intent)
        else:
            self._queued.append(intent)

    def on_start(self) -> None:
        self._ready = True
        queued, self._queued = self._queued, []
        for intent in queued:
            self._submit(intent)

    def on_stop(self) -> None:
        self._ready = False

    def cancel_strategy_orders(self, strategy_id: str) -> None:
        self._queued = [item for item in self._queued if item.strategy_id != strategy_id]
        for client_order_id, intent in tuple(self._intents.items()):
            if intent.strategy_id != strategy_id:
                continue
            native_order = self.cache.order(client_order_id)
            if native_order is not None and native_order.is_open:
                self.cancel_order(native_order)

    def _submit(self, intent: OrderIntent) -> None:
        instrument = self.cache.instrument(intent.instrument_id)
        if instrument is None:
            raise RuntimeError(f"找不到模拟合约: {intent.instrument_id}")
        native_side = (
            NativeOrderSide.BUY if intent.side is OrderSide.BUY else NativeOrderSide.SELL
        )
        quantity = instrument.make_qty(intent.quantity)
        reducing = intent.reduce_only or intent.position_effect in {
            PositionEffect.CLOSE,
            PositionEffect.CLOSE_TODAY,
            PositionEffect.CLOSE_YESTERDAY,
        }
        if intent.position_effect is PositionEffect.OPEN and intent.reduce_only:
            raise ValueError("OPEN订单不能同时设置reduce_only")

        if intent.order_type is OrderType.MARKET:
            native_order = self.order_factory.market(
                instrument_id=intent.instrument_id,
                order_side=native_side,
                quantity=quantity,
                reduce_only=reducing,
            )
        else:
            if intent.price is None:  # OrderIntent自身已经校验，此处保护类型收窄。
                raise ValueError("限价单缺少价格")
            native_order = self.order_factory.limit(
                instrument_id=intent.instrument_id,
                order_side=native_side,
                quantity=quantity,
                price=instrument.make_price(intent.price),
                reduce_only=reducing,
            )

        self._intents[native_order.client_order_id] = intent
        self.submit_order(native_order)

    def on_order_accepted(self, event: OrderAccepted) -> None:
        intent = self._intents.get(event.client_order_id)
        if intent is None:
            return
        self._emit(intent, event.client_order_id, ExecutionReportType.ACCEPTED, event.ts_event)

    def on_order_filled(self, event: OrderFilled) -> None:
        intent = self._intents.get(event.client_order_id)
        if intent is None:
            return
        native_order = self.cache.order(event.client_order_id)
        closed = native_order is not None and native_order.is_closed
        self._emit(
            intent,
            event.client_order_id,
            ExecutionReportType.FILLED if closed else ExecutionReportType.PARTIALLY_FILLED,
            event.ts_event,
            filled_quantity=Decimal(str(event.last_qty)),
            fill_price=Decimal(str(event.last_px)),
            metadata={
                "trade_id": str(event.trade_id),
                "venue_order_id": str(event.venue_order_id),
                "commission": str(event.commission),
            },
        )
        if closed:
            self._intents.pop(event.client_order_id, None)

    def on_order_rejected(self, event: OrderRejected) -> None:
        self._finish(
            event.client_order_id,
            event.ts_event,
            ExecutionReportType.REJECTED,
            event.reason,
        )

    def on_order_denied(self, event: OrderDenied) -> None:
        self._finish(
            event.client_order_id,
            event.ts_event,
            ExecutionReportType.REJECTED,
            event.reason,
        )

    def on_order_canceled(self, event: OrderCanceled) -> None:
        self._finish(event.client_order_id, event.ts_event, ExecutionReportType.CANCELED, None)

    def on_order_expired(self, event: OrderExpired) -> None:
        self._finish(event.client_order_id, event.ts_event, ExecutionReportType.CANCELED, "expired")

    def _finish(
        self,
        client_order_id: ClientOrderId,
        ts_event: int,
        report_type: ExecutionReportType,
        reason: object | None,
    ) -> None:
        intent = self._intents.pop(client_order_id, None)
        if intent is None:
            return
        self._emit(
            intent,
            client_order_id,
            report_type,
            ts_event,
            reason=None if reason is None else str(reason),
        )

    def _emit(
        self,
        intent: OrderIntent,
        client_order_id: ClientOrderId,
        report_type: ExecutionReportType,
        ts_event: int,
        *,
        filled_quantity: Decimal = Decimal(0),
        fill_price: Decimal | None = None,
        reason: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        report_metadata = {
            "strategy_id": intent.strategy_id,
            "position_effect": intent.position_effect.value,
            **dict(intent.metadata),
            **(metadata or {}),
        }
        self._report_sink(
            ExecutionReport(
                backend_id=self._backend_id,
                client_order_id=str(client_order_id),
                instrument_id=intent.instrument_id,
                report_type=report_type,
                ts_event=ts_event,
                filled_quantity=filled_quantity,
                fill_price=fill_price,
                order_side=intent.side,
                order_quantity=intent.quantity,
                position_effect=intent.position_effect,
                reason=reason,
                metadata=report_metadata,
            ),
        )
