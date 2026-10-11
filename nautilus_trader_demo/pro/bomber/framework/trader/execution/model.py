"""独立策略参考价执行模型；模型成交始终标记来源，不连接物理账户。"""

from uuid import uuid4
from .contracts import ExecutionBackendKind, ExecutionReport, ExecutionReportType, OrderType


class ReferencePriceExecutionBackend:
    kind = ExecutionBackendKind.SIMULATION

    def __init__(self, backend_id, prices):
        self.backend_id, self.prices = backend_id, prices
        self._started = False
        self._handlers, self._reports = [], []

    @property
    def reports(self):
        return tuple(self._reports)

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def register_report_handler(self, handler):
        if handler not in self._handlers:
            self._handlers.append(handler)

    def submit_order(self, order):
        if not self._started or order.backend_id != self.backend_id:
            raise RuntimeError("模型执行端未启动或订单路由不符")
        reference = self.prices.get(order.instrument_id)
        if reference is None or order.order_type is not OrderType.MARKET:
            raise ValueError("参考价模型仅接受有参考价的市价意图")
        ts = int(order.metadata.get("request_ts", reference.ts_event))
        if reference.ts_event > ts or reference.price <= 0:
            raise ValueError("模型不得使用未来或无效参考价")
        order_id = f"model-{uuid4().hex}"
        for sequence, status in enumerate((ExecutionReportType.ACCEPTED, ExecutionReportType.FILLED), 1):
            report = ExecutionReport(self.backend_id, order_id, order.instrument_id, status, ts,
                filled_quantity=order.quantity if status is ExecutionReportType.FILLED else 0,
                fill_price=reference.price if status is ExecutionReportType.FILLED else None,
                order_side=order.side, order_quantity=order.quantity, position_effect=order.position_effect,
                report_id=f"{order_id}:{sequence}", sequence=sequence,
                metadata={**order.metadata, "strategy_id": order.strategy_id,
                          "trade_id": f"{order_id}:trade" if status is ExecutionReportType.FILLED else None,
                          "execution_origin": "model", "model": "reference-price"})
            self._reports.append(report)
            for handler in tuple(self._handlers):
                handler(report)

    def cancel_strategy(self, strategy_id):
        # 该显式模型同步成交，没有真实活动委托。
        pass
