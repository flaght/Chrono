"""把Runner迁移期ExecutionClientPort连接到Planner和Backend。"""

from __future__ import annotations

from decimal import Decimal

from strategy.contracts import ExecutionRequest
from strategy.execution.contracts import ExecutionReport, ExecutionReportType, OrderSide
from strategy.execution.ports import ExecutionBackendPort, OrderPlannerPort
from strategy.portfolio import PositionManager


class BackendExecutionClient:
    """目标请求 → Planner → OrderIntent → Backend的组合适配器。"""

    def __init__(
        self,
        client_id: str,
        planner: OrderPlannerPort,
        backend: ExecutionBackendPort,
        position_manager: PositionManager,
    ) -> None:
        if client_id != backend.backend_id:
            raise ValueError("client_id必须与backend_id一致")
        self.client_id = client_id
        self.planner = planner
        self.backend = backend
        self.position_manager = position_manager
        self._started = False
        self._remaining_by_order_id: dict[str, Decimal] = {}
        self.backend.register_report_handler(self._on_report)

    def start(self) -> None:
        if self._started:
            return
        self.backend.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self.backend.stop()
        self._started = False

    def submit_targets(self, request: ExecutionRequest) -> None:
        if request.client_id != self.client_id:
            raise ValueError("ExecutionRequest客户端不匹配")
        for order in self.planner.plan(request):
            signed = order.quantity if order.side is OrderSide.BUY else -order.quantity
            self.position_manager.adjust_working_quantity(
                self.client_id,
                order.instrument_id,
                signed,
            )
            try:
                self.backend.submit_order(order)
            except Exception:
                self.position_manager.adjust_working_quantity(
                    self.client_id,
                    order.instrument_id,
                    -signed,
                )
                raise

    def cancel_strategy(self, strategy_id: str) -> None:
        self.backend.cancel_strategy(strategy_id)

    def _on_report(self, report: ExecutionReport) -> None:
        if report.order_side is None:
            return
        direction = Decimal(1) if report.order_side is OrderSide.BUY else Decimal(-1)
        if (
            report.report_type is ExecutionReportType.ACCEPTED
            and report.order_quantity is not None
        ):
            self._remaining_by_order_id.setdefault(
                report.client_order_id,
                report.order_quantity,
            )
        if report.filled_quantity:
            signed_fill = direction * report.filled_quantity
            self.position_manager.adjust_account_position(
                self.client_id,
                report.instrument_id,
                signed_fill,
            )
            self.position_manager.adjust_working_quantity(
                self.client_id,
                report.instrument_id,
                -signed_fill,
            )
            if report.client_order_id in self._remaining_by_order_id:
                remaining = self._remaining_by_order_id[report.client_order_id]
                self._remaining_by_order_id[report.client_order_id] = max(
                    Decimal(0),
                    remaining - report.filled_quantity,
                )
            if report.report_type is ExecutionReportType.FILLED:
                self._remaining_by_order_id.pop(report.client_order_id, None)
        if report.report_type in {
            ExecutionReportType.REJECTED,
            ExecutionReportType.CANCELED,
        } and report.order_quantity is not None:
            # 已有Accepted/Partial回报时释放本地剩余量；直接Rejected时回退到
            # 原始订单数量，防止部分成交后撤单把在途数量多释放一次。
            remaining_quantity = self._remaining_by_order_id.pop(
                report.client_order_id,
                report.order_quantity,
            )
            remaining = direction * remaining_quantity
            self.position_manager.adjust_working_quantity(
                self.client_id,
                report.instrument_id,
                -remaining,
            )
