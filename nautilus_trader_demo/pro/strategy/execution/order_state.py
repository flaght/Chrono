"""ExecutionReport幂等处理和订单生命周期状态机。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import threading
from typing import Sequence

from market.basic.base import InstrumentId
from strategy.execution.contracts import (
    ExecutionReport,
    ExecutionReportType,
    OrderSide,
)


class OrderStateError(RuntimeError):
    """执行回报与已知订单状态冲突，不能安全地继续更新仓位。"""


class OrderLifecycleStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            OrderLifecycleStatus.FILLED,
            OrderLifecycleStatus.CANCELED,
            OrderLifecycleStatus.REJECTED,
        }


@dataclass(frozen=True)
class OrderState:
    """一个客户端订单对外可见的不可变状态。"""

    backend_id: str
    client_order_id: str
    instrument_id: InstrumentId
    side: OrderSide
    order_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    status: OrderLifecycleStatus
    last_sequence: int | None
    last_ts_event: int


@dataclass(frozen=True)
class OrderReportUpdate:
    """状态机对单份回报的处理结果，数量均为无符号增量。"""

    applied: bool
    fill_delta: Decimal
    release_delta: Decimal
    state: OrderState


@dataclass(frozen=True)
class OrderStateCheckpoint:
    """包含幂等键的可持久化订单状态。"""

    state: OrderState
    seen_keys: tuple[str, ...] = ()


@dataclass
class _OrderRecord:
    backend_id: str
    client_order_id: str
    instrument_id: InstrumentId
    side: OrderSide
    order_quantity: Decimal
    filled_quantity: Decimal = Decimal(0)
    status: OrderLifecycleStatus = OrderLifecycleStatus.PENDING
    last_sequence: int | None = None
    last_ts_event: int = 0
    seen_keys: set[str] = field(default_factory=set)

    def snapshot(self) -> OrderState:
        return OrderState(
            backend_id=self.backend_id,
            client_order_id=self.client_order_id,
            instrument_id=self.instrument_id,
            side=self.side,
            order_quantity=self.order_quantity,
            filled_quantity=self.filled_quantity,
            remaining_quantity=self.order_quantity - self.filled_quantity,
            status=self.status,
            last_sequence=self.last_sequence,
            last_ts_event=self.last_ts_event,
        )


class OrderReportStateMachine:
    """按订单去重、排序并计算成交和剩余委托释放量。"""

    def __init__(self, backend_id: str) -> None:
        if not backend_id.strip():
            raise ValueError("backend_id不能为空")
        self.backend_id = backend_id
        self._orders: dict[str, _OrderRecord] = {}
        self._lock = threading.RLock()

    def state(self, client_order_id: str) -> OrderState | None:
        with self._lock:
            record = self._orders.get(client_order_id)
            return None if record is None else record.snapshot()

    def states(self) -> tuple[OrderState, ...]:
        with self._lock:
            return tuple(record.snapshot() for record in self._orders.values())

    def checkpoints(self) -> tuple[OrderStateCheckpoint, ...]:
        with self._lock:
            return tuple(
                OrderStateCheckpoint(record.snapshot(), tuple(sorted(record.seen_keys)))
                for record in self._orders.values()
            )

    def restore(self, checkpoints: Sequence[OrderStateCheckpoint]) -> None:
        restored: dict[str, _OrderRecord] = {}
        for checkpoint in checkpoints:
            state = checkpoint.state
            if state.backend_id != self.backend_id:
                raise OrderStateError("恢复订单的backend_id不匹配")
            if state.client_order_id in restored:
                raise OrderStateError("恢复快照包含重复client_order_id")
            if state.filled_quantity < 0 or state.remaining_quantity < 0:
                raise OrderStateError("恢复订单数量不能为负数")
            if state.filled_quantity + state.remaining_quantity != state.order_quantity:
                raise OrderStateError("恢复订单的成交量与剩余量不守恒")
            restored[state.client_order_id] = _OrderRecord(
                backend_id=state.backend_id,
                client_order_id=state.client_order_id,
                instrument_id=state.instrument_id,
                side=state.side,
                order_quantity=state.order_quantity,
                filled_quantity=state.filled_quantity,
                status=state.status,
                last_sequence=state.last_sequence,
                last_ts_event=state.last_ts_event,
                seen_keys=set(checkpoint.seen_keys),
            )
        with self._lock:
            self._orders = restored

    def apply(self, report: ExecutionReport) -> OrderReportUpdate:
        with self._lock:
            return self._apply_unlocked(report)

    def _apply_unlocked(self, report: ExecutionReport) -> OrderReportUpdate:
        if report.backend_id != self.backend_id:
            raise OrderStateError(
                f"回报Backend不匹配: {report.backend_id} != {self.backend_id}",
            )
        record = self._orders.get(report.client_order_id)
        if record is None:
            if report.order_side is None or report.order_quantity is None:
                raise OrderStateError("首份回报缺少order_side或order_quantity")
            record = _OrderRecord(
                backend_id=report.backend_id,
                client_order_id=report.client_order_id,
                instrument_id=report.instrument_id,
                side=report.order_side,
                order_quantity=report.order_quantity,
            )
            self._orders[report.client_order_id] = record
        self._validate_identity(record, report)

        report_key = self._report_key(report)
        if report_key in record.seen_keys:
            return self._ignored(record)
        if (
            report.sequence is not None
            and record.last_sequence is not None
            and report.sequence <= record.last_sequence
        ):
            return self._ignored(record)
        if (
            report.sequence is None
            and record.last_ts_event
            and report.ts_event < record.last_ts_event
        ):
            return self._ignored(record)
        if record.status.is_terminal:
            return self._ignored(record)

        fill_delta = Decimal(0)
        release_delta = Decimal(0)
        next_status = record.status
        if report.report_type is ExecutionReportType.ACCEPTED:
            if record.status is OrderLifecycleStatus.PENDING:
                next_status = OrderLifecycleStatus.ACCEPTED
        elif report.report_type in {
            ExecutionReportType.PARTIALLY_FILLED,
            ExecutionReportType.FILLED,
        }:
            if report.filled_quantity <= 0:
                raise OrderStateError("成交回报的filled_quantity必须大于零")
            new_filled = record.filled_quantity + report.filled_quantity
            if new_filled > record.order_quantity:
                raise OrderStateError(
                    f"订单过量成交: order={record.order_quantity} filled={new_filled}",
                )
            if (
                report.report_type is ExecutionReportType.FILLED
                and new_filled != record.order_quantity
            ):
                raise OrderStateError(
                    f"FILLED回报与累计成交不一致: order={record.order_quantity} "
                    f"filled={new_filled}",
                )
            fill_delta = report.filled_quantity
            record.filled_quantity = new_filled
            next_status = (
                OrderLifecycleStatus.FILLED
                if new_filled == record.order_quantity
                else OrderLifecycleStatus.PARTIALLY_FILLED
            )
        elif report.report_type in {
            ExecutionReportType.CANCELED,
            ExecutionReportType.REJECTED,
        }:
            release_delta = record.order_quantity - record.filled_quantity
            next_status = (
                OrderLifecycleStatus.CANCELED
                if report.report_type is ExecutionReportType.CANCELED
                else OrderLifecycleStatus.REJECTED
            )

        record.status = next_status
        record.last_sequence = (
            report.sequence if report.sequence is not None else record.last_sequence
        )
        record.last_ts_event = max(record.last_ts_event, report.ts_event)
        record.seen_keys.add(report_key)
        return OrderReportUpdate(
            applied=True,
            fill_delta=fill_delta,
            release_delta=release_delta,
            state=record.snapshot(),
        )

    @staticmethod
    def _validate_identity(record: _OrderRecord, report: ExecutionReport) -> None:
        if report.instrument_id != record.instrument_id:
            raise OrderStateError("同一client_order_id的instrument_id发生变化")
        if report.order_side is not None and report.order_side is not record.side:
            raise OrderStateError("同一client_order_id的order_side发生变化")
        if (
            report.order_quantity is not None
            and report.order_quantity != record.order_quantity
        ):
            raise OrderStateError("同一client_order_id的order_quantity发生变化")

    @staticmethod
    def _report_key(report: ExecutionReport) -> str:
        if report.report_id is not None:
            return f"report_id:{report.report_id}"
        if report.sequence is not None:
            return f"sequence:{report.sequence}"
        return "|".join(
            (
                "fallback",
                report.report_type.value,
                str(report.ts_event),
                str(report.filled_quantity),
                str(report.fill_price),
                str(report.reason),
            ),
        )

    @staticmethod
    def _ignored(record: _OrderRecord) -> OrderReportUpdate:
        return OrderReportUpdate(
            applied=False,
            fill_delta=Decimal(0),
            release_delta=Decimal(0),
            state=record.snapshot(),
        )
