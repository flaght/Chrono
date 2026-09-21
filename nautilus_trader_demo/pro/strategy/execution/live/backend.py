"""由可替换Nautilus实时Driver驱动的统一Live Backend。"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Callable

from market.basic.base import InstrumentId
from strategy.execution.contracts import (
    AccountPositionSnapshot,
    ExecutionBackendKind,
    ExecutionReport,
    OrderIntent,
)
from strategy.execution.live.driver import NautilusLiveDriverPort


class NautilusLiveExecutionBackend:
    """把Nautilus TradingNode生命周期隐藏在ExecutionBackendPort之后。"""

    kind = ExecutionBackendKind.LIVE

    def __init__(self, backend_id: str, driver: NautilusLiveDriverPort) -> None:
        if not backend_id.strip():
            raise ValueError("backend_id不能为空")
        if driver.driver_id != backend_id:
            raise ValueError("driver_id必须与backend_id一致")
        self.backend_id = backend_id
        self.driver = driver
        self._report_handlers: list[Callable[[ExecutionReport], None]] = []
        self._reports: list[ExecutionReport] = []
        self._started = False
        self._reconcile_revision = 0

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def reports(self) -> tuple[ExecutionReport, ...]:
        return tuple(self._reports)

    def start(self) -> None:
        if self._started:
            return
        self.driver.start(self._receive_report)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.driver.stop()
        finally:
            self._started = False

    def submit_order(self, order: OrderIntent) -> None:
        if not self._started:
            raise RuntimeError("Live Backend尚未启动")
        if order.backend_id != self.backend_id:
            raise ValueError(
                f"订单Backend不匹配: {order.backend_id} != {self.backend_id}",
            )
        self.driver.submit_order(order)

    def register_report_handler(
        self,
        handler: Callable[[ExecutionReport], None],
    ) -> None:
        if handler not in self._report_handlers:
            self._report_handlers.append(handler)

    def cancel_strategy(self, strategy_id: str) -> None:
        if not strategy_id.strip():
            raise ValueError("strategy_id不能为空")
        self.driver.cancel_strategy(strategy_id)

    def reconcile(self) -> AccountPositionSnapshot:
        if not self._started:
            raise RuntimeError("Live Backend尚未启动")
        raw_positions = self.driver.reconcile()
        positions: dict[InstrumentId, Decimal] = {}
        for instrument_id, quantity in raw_positions.items():
            normalized_id = (
                instrument_id
                if isinstance(instrument_id, InstrumentId)
                else InstrumentId.from_str(str(instrument_id))
            )
            positions[normalized_id] = (
                quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
            )
        self._reconcile_revision += 1
        return AccountPositionSnapshot(
            backend_id=self.backend_id,
            revision=self._reconcile_revision,
            ts_event=time.time_ns(),
            positions=positions,
        )

    def _receive_report(self, report: ExecutionReport) -> None:
        if report.backend_id != self.backend_id:
            raise ValueError(
                f"执行回报Backend不匹配: {report.backend_id} != {self.backend_id}",
            )
        self._reports.append(report)
        for handler in tuple(self._report_handlers):
            handler(report)
