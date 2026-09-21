"""把Runner迁移期ExecutionClientPort连接到Planner和Backend。"""

from __future__ import annotations

import threading

from strategy.contracts import ExecutionRequest
from strategy.execution.contracts import ExecutionReport, OrderSide
from strategy.execution.order_state import (
    OrderReportStateMachine,
    OrderState,
    OrderStateCheckpoint,
    OrderStateError,
)
from strategy.execution.ports import LiveExecutionBackendPort, OrderPlannerPort
from strategy.execution.risk import KillSwitchMode, PreTradeRiskManager
from strategy.portfolio import PositionManager


class BackendExecutionClient:
    """目标请求 → Planner → OrderIntent → Backend的组合适配器。"""

    def __init__(
        self,
        client_id: str,
        planner: OrderPlannerPort,
        backend: LiveExecutionBackendPort,
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
        self._reconciled = False
        self._order_states = OrderReportStateMachine(client_id)
        self._report_errors: list[OrderStateError] = []
        self._strategy_ids: set[str] = set()
        self._submit_lock = threading.RLock()
        self.backend.register_report_handler(self._on_report)

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
                except Exception:
                    self.position_manager.adjust_working_quantity(
                        self.client_id,
                        order.instrument_id,
                        -signed,
                    )
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
