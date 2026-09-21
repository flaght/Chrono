"""统一Runner使用的模拟执行客户端。"""

from __future__ import annotations

import threading

from strategy.contracts import ExecutionRequest
from strategy.execution.contracts import ExecutionReport, OrderSide
from strategy.execution.order_state import (
    OrderReportStateMachine,
    OrderState,
    OrderStateError,
)
from strategy.execution.ports import OrderPlannerPort, SimExecutionBackendPort
from strategy.execution.risk import KillSwitchMode, PreTradeRiskManager
from strategy.portfolio import PositionManager


class SimulationExecutionClient:
    """目标请求经Planner/Risk进入模拟Backend，并用统一回报维护仓位。

    与Live客户端的区别只在启动边界：模拟账户由Backend创建，不执行柜台权威
    仓位对账；目标规划、前置风控、在途量和ExecutionReport处理保持同一语义。
    """

    def __init__(
        self,
        client_id: str,
        planner: OrderPlannerPort,
        backend: SimExecutionBackendPort,
        position_manager: PositionManager,
        risk_manager: PreTradeRiskManager | None = None,
    ) -> None:
        if client_id != backend.backend_id:
            raise ValueError("client_id必须与backend_id一致")
        self.client_id = client_id
        self.planner = planner
        self.backend = backend
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self._started = False
        self._order_states = OrderReportStateMachine(client_id)
        self._report_errors: list[OrderStateError] = []
        self._submit_lock = threading.RLock()
        self.backend.register_report_handler(self._on_report)

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def report_errors(self) -> tuple[OrderStateError, ...]:
        return tuple(self._report_errors)

    @property
    def order_state_machine(self) -> OrderReportStateMachine:
        return self._order_states

    def order_state(self, client_order_id: str) -> OrderState | None:
        return self._order_states.state(client_order_id)

    def start(self) -> None:
        if self._started:
            return
        self.backend.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.backend.stop()
        finally:
            self._started = False

    def submit_targets(self, request: ExecutionRequest) -> None:
        with self._submit_lock:
            if not self._started:
                raise RuntimeError("模拟执行客户端尚未启动")
            if request.client_id != self.client_id:
                raise ValueError("ExecutionRequest客户端不匹配")
            if self._report_errors:
                raise RuntimeError("模拟执行回报存在状态冲突，拒绝继续提交目标")
            orders = tuple(self.planner.plan(request))
            health_reduce_only = (
                request.metadata.get("market_health_mode")
                == KillSwitchMode.REDUCE_ONLY.value
            )
            if self.risk_manager is not None:
                # 内部Tick聚合的Bar时间可能是上一分钟边界，而触发该Bar的
                # 下一分钟Quote/Trade已先进入参考价存储。风控评估时刻应是
                # 当前已经处理到的行情时刻，不能用较早的信号Bar时间把有效
                # 参考价误判为“未来行情”。ExecutionRequest.ts_event仍保留
                # 原始信号时间，供回测前视审计。
                reference_times = (
                    reference.ts_event
                    for order in orders
                    if (reference := self.risk_manager.price_store.get(order.instrument_id))
                    is not None
                )
                # 单个时间戳不能调用max(value)：那会进入「单可迭代对象」重载，
                # 在尚无参考行情（或本批无订单）时对int抛TypeError。
                # 将信号时间也放入序列，保证空reference_times仍有一个候选值。
                self.risk_manager.check(
                    orders,
                    now_ns=max((request.ts_event, *reference_times)),
                    mode_override=(
                        KillSwitchMode.REDUCE_ONLY
                        if health_reduce_only
                        else None
                    ),
                )
            elif health_reduce_only and orders:
                raise RuntimeError(
                    "行情降级期间模拟执行必须配置PreTradeRiskManager",
                )

            for order in orders:
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
        try:
            update = self._order_states.apply(report)
        except OrderStateError as error:
            self._report_errors.append(error)
            return
        if not update.applied:
            return
        direction = 1 if update.state.side is OrderSide.BUY else -1
        if update.fill_delta:
            signed_fill = direction * update.fill_delta
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
        if update.release_delta:
            remaining = direction * update.release_delta
            self.position_manager.adjust_working_quantity(
                self.client_id,
                report.instrument_id,
                -remaining,
            )
