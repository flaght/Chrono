"""把Runner迁移期ExecutionClientPort连接到Planner和Backend。"""

from __future__ import annotations

import threading
from typing import Callable

from trader.contracts import ExecutionRequest
from trader.execution.contracts import ExecutionReport, OrderSide
from trader.execution.events import FillEvent, OrderUpdateEvent, map_applied_report
from trader.execution.order_state import (
    OrderReportStateMachine,
    OrderState,
    OrderStateCheckpoint,
    OrderStateError,
)
from trader.execution.ports import LiveExecutionBackendPort, OrderPlannerPort
from trader.execution.risk import KillSwitchMode, PreTradeRiskManager
from trader.portfolio import PositionManager


class BackendExecutionClient:
    """目标请求 → Planner → OrderIntent → Backend的组合适配器。"""

    def __init__(
        self,
        client_id: str,
        planner: OrderPlannerPort,
        backend: LiveExecutionBackendPort,
        position_manager: PositionManager,
        risk_manager: PreTradeRiskManager | None = None,
        *,
        account_id: str | None = None,
    ) -> None:
        if client_id != backend.backend_id:
            raise ValueError("client_id必须与backend_id一致")
        self.client_id = client_id
        # 旧接口默认一个client对应一个账户；多账户适配器必须显式给出账户ID。
        self.account_id = account_id if account_id is not None else client_id
        if not isinstance(self.account_id, str) or not self.account_id.strip():
            raise ValueError("account_id不能为空")
        self.planner = planner
        self.backend = backend
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self._started = False
        self._reconciled = False
        self._order_states = OrderReportStateMachine(client_id)
        self._report_errors: list[OrderStateError] = []
        self._strategy_ids: set[str] = set()
        self._submit_lock = threading.RLock()
        self._execution_handlers: list[Callable[[OrderUpdateEvent | FillEvent], None]] = []
        self._submit_failure_handler: Callable[[], None] | None = None
        self.backend.register_report_handler(self._on_report)

    def register_submit_failure_handler(self, handler: Callable[[], None]) -> None:
        """发送请求同步失败、撤回在途量后保存修正后的状态。"""
        if self._started or self._submit_failure_handler is not None:
            raise RuntimeError("提交失败处理器只能在客户端启动前绑定一次")
        self._submit_failure_handler = handler

    def register_execution_event_handler(
        self, handler: Callable[[OrderUpdateEvent | FillEvent], None],
    ) -> None:
        """可选能力；旧ExecutionClientPort与既有调用方无需修改。"""
        if handler not in self._execution_handlers:
            self._execution_handlers.append(handler)

    def start(self) -> None:
        if self._started:
            return
        self.backend.start()
        try:
            self._started = True
            self.reconcile()
        except Exception:
            self._started = False
            self._reconciled = False
            self.position_manager.clear_account_reconciliation(self.client_id)
            self.backend.stop()
            raise

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.backend.stop()
        finally:
            self._started = False
            self._reconciled = False
            self.position_manager.clear_account_reconciliation(self.client_id)

    @property
    def is_reconciled(self) -> bool:
        return self._reconciled

    @property
    def report_errors(self) -> tuple[OrderStateError, ...]:
        return tuple(self._report_errors)

    def order_state(self, client_order_id: str) -> OrderState | None:
        return self._order_states.state(client_order_id)

    @property
    def order_state_machine(self) -> OrderReportStateMachine:
        return self._order_states

    def restore_order_checkpoints(
        self,
        checkpoints: tuple[OrderStateCheckpoint, ...],
    ) -> None:
        if self._started:
            raise RuntimeError("执行客户端启动后不能恢复订单快照")
        self._order_states.restore(checkpoints)

    def reconcile(self) -> None:
        if not self._started:
            raise RuntimeError("执行客户端尚未启动，不能对账")
        if self._report_errors:
            raise RuntimeError(
                "存在执行回报状态冲突；仅账户仓位对账不足以恢复活动订单状态",
            )
        self._reconciled = False
        snapshot = self.backend.reconcile()
        if snapshot.backend_id != self.client_id:
            raise ValueError("权威仓位快照Backend不匹配")
        self.position_manager.apply_account_snapshot(
            self.client_id,
            snapshot.positions,
            revision=snapshot.revision,
            ts_event=snapshot.ts_event,
        )
        if self.position_manager.is_recovery_required(self.client_id):
            self.position_manager.clear_account_reconciliation(self.client_id)
            raise RuntimeError(
                "检测到重启前活动订单；必须先用柜台查询结果完成在途订单恢复",
            )
        self._reconciled = True

    def submit_targets(self, request: ExecutionRequest) -> None:
        with self._submit_lock:
            if not self._started or not self._reconciled:
                raise RuntimeError("账户尚未完成权威仓位对账，拒绝下单")
            if request.client_id != self.client_id:
                raise ValueError("ExecutionRequest客户端不匹配")
            orders = tuple(self.planner.plan(request))
            health_reduce_only = (
                request.metadata.get("market_health_mode")
                == KillSwitchMode.REDUCE_ONLY.value
            )
            if self.risk_manager is not None:
                self.risk_manager.check(
                    orders,
                    now_ns=request.ts_event,
                    mode_override=(
                        KillSwitchMode.REDUCE_ONLY
                        if health_reduce_only
                        else None
                    ),
                )
            elif health_reduce_only and orders:
                raise RuntimeError(
                    "行情降级期间Live执行必须配置PreTradeRiskManager以校验真实仓位",
                )
            self._strategy_ids.add(request.strategy_id)
            for order in orders:
                signed = order.quantity if order.side is OrderSide.BUY else -order.quantity
                self.position_manager.adjust_working_quantity(
                    self.client_id,
                    order.instrument_id,
                    signed,
                )
                try:
                    self.backend.submit_order(order)
                except Exception as error:
                    if getattr(error, "order_may_be_live", False):
                        self._reconciled = False
                        self.position_manager.mark_recovery_required(self.client_id)
                        raise
                    self.position_manager.adjust_working_quantity(
                        self.client_id,
                        order.instrument_id,
                        -signed,
                    )
                    if self._submit_failure_handler is not None:
                        try:
                            self._submit_failure_handler()
                        except Exception:
                            self._reconciled = False
                            self.position_manager.clear_account_reconciliation(self.client_id)
                            raise
                    raise

    def set_risk_mode(
        self,
        mode: KillSwitchMode | str,
        *,
        cancel_active_orders: bool = True,
    ) -> None:
        if self.risk_manager is None:
            raise RuntimeError("执行客户端未配置PreTradeRiskManager")
        normalized = KillSwitchMode(mode)
        self.risk_manager.set_mode(normalized)
        if normalized is not KillSwitchMode.NORMAL and cancel_active_orders:
            for strategy_id in tuple(self._strategy_ids):
                self.backend.cancel_strategy(strategy_id)

    def cancel_strategy(self, strategy_id: str) -> None:
        self.backend.cancel_strategy(strategy_id)

    def _on_report(self, report: ExecutionReport) -> None:
        # 状态机先接受并更新仓位/在途，然后才允许对策略投递事件。
        # 旧调用方未订阅事件时保持原有的回报处理路径。
        with self._submit_lock:
            self._apply_report(report)

    def _apply_report(self, report: ExecutionReport) -> None:
        previous = self._order_states.state(report.client_order_id)
        try:
            update = self._order_states.apply(report)
        except OrderStateError as error:
            # 状态冲突时不能猜测仓位或剩余委托；立即关闭下单闸门，等待人工
            # 检查和更完整的活动订单对账。
            self._report_errors.append(error)
            self._reconciled = False
            self.position_manager.clear_account_reconciliation(self.client_id)
            return
        if not update.applied:
            return
        changed = (
            previous is None
            or previous.status is not update.state.status
            or previous.filled_quantity != update.state.filled_quantity
        )
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
        if not self._execution_handlers or not changed:
            return
        try:
            order_event, fill_event = map_applied_report(
                report, update, account_id=self.account_id,
            )
            for event in (order_event, fill_event):
                if event is None:
                    continue
                for handler in tuple(self._execution_handlers):
                    handler(event)
        except Exception as error:
            # 已被状态机接受的回报不能回滚；关联失败/回调异常必须关闭闸门，
            # 等待账户和活动订单重新对账，而不是静默丢失策略通知。
            self._report_errors.append(OrderStateError(f"标准执行事件分发失败: {error}"))
            self._reconciled = False
            self.position_manager.clear_account_reconciliation(self.client_id)
