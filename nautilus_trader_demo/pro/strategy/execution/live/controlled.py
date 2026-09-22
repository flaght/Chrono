"""P3受控在线执行样板：默认只读，显式DEMO授权后才可下单。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from market.basic.base import InstrumentId
from strategy.contracts import ExecutionRequest
from strategy.execution.contracts import ExecutionReport
from strategy.execution.events import AccountStateEvent, ExecutionSide
from strategy.execution.live.client import BackendExecutionClient
from strategy.execution.risk import KillSwitchMode


@dataclass(frozen=True)
class LiveAuditRecord:
    ts_ns: int
    action: str
    detail: str


class ControlledLiveExecutionClient(BackendExecutionClient):
    """与普通Live客户端同一端口，但要求权威仓位和资金对账及人工授权。

    `demo_environment_check`必须从实际TradingNode/账户配置确认是DEMO，不能仅凭
    调用方传入一个布尔标志。此类本身不连接网络，也不自动发送测试订单。
    """

    DEMO_CONFIRMATION = "AUTHORIZE_DEMO_ORDERS"

    def __init__(
        self,
        *args,
        demo_environment_check: Callable[[], bool],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.risk_manager is None:
            raise ValueError("P3受控在线客户端必须配置PreTradeRiskManager")
        self._demo_environment_check = demo_environment_check
        self._armed = False
        self._account_state: AccountStateEvent | None = None
        self._last_account_revision = 0
        self._last_orders_revision = 0
        self._orders_recovery_pending = self.position_manager.is_recovery_required(self.client_id)
        self._connection_epoch = 0
        self._heartbeat_in_flight = False
        self._audit: list[LiveAuditRecord] = []

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def account_state(self) -> AccountStateEvent | None:
        return self._account_state

    @property
    def audit_records(self) -> tuple[LiveAuditRecord, ...]:
        return tuple(self._audit)

    def _record(self, action: str, detail: str = "") -> None:
        self._audit.append(LiveAuditRecord(time.time_ns(), action, detail))

    def start(self) -> None:
        if self._started:
            return
        self.backend.start()
        try:
            self._started = True
            if self._orders_recovery_pending or self.position_manager.is_recovery_required(self.client_id):
                self.position_manager.mark_recovery_required(self.client_id)
                self.recover_active_orders()
            self.reconcile()
        except Exception:
            self._started = False
            self._reconciled = False
            self._account_state = None
            self.position_manager.clear_account_reconciliation(self.client_id)
            self.backend.stop()
            raise

    def recover_active_orders(self) -> None:
        """仅在柜台快照与本地已知活动订单完全一致时恢复在途量。

        新订单、断线期间新增成交或消失的旧订单需要更完整的成交/订单状态对账；
        本方法遇到这些情况保持闭闸，不能把未知订单当作撤销。
        """
        with self._submit_lock:
            if not self._started or not (
                self._orders_recovery_pending
                or self.position_manager.is_recovery_required(self.client_id)
            ):
                raise RuntimeError("客户端未启动或不需要活动订单恢复")
            # 外部误清除PositionManager标志也不能跳过柜台订单验证。
            self.position_manager.mark_recovery_required(self.client_id)
            if self.report_errors:
                raise RuntimeError("执行回报冲突尚未处理，不能自动恢复活动订单")
            self._armed = False
            self._reconciled = False
            self._account_state = None
            epoch = self._connection_epoch
            known = {
                state.client_order_id: state
                for state in self.order_state_machine.states()
                if not state.status.is_terminal
            }
        try:
            snapshot = self.backend.reconcile_active_orders()
            with self._submit_lock:
                if epoch != self._connection_epoch or self.report_errors:
                    raise RuntimeError("活动订单查询期间发生断线或回报冲突")
                if snapshot.account_id != self.account_id:
                    raise ValueError("活动订单快照account_id与执行账户不匹配")
                if snapshot.revision <= self._last_orders_revision:
                    raise ValueError("活动订单快照版本未单调递增")
                current = {
                    state.client_order_id: state
                    for state in self.order_state_machine.states()
                    if not state.status.is_terminal
                }
                if current != known:
                    raise RuntimeError("查询期间本地活动订单状态变化")
                reported = {order.client_order_id: order for order in snapshot.orders}
                if reported.keys() != known.keys():
                    raise RuntimeError("柜台与本地活动订单ID不一致，需要人工对账")
                working: dict[InstrumentId, Decimal] = {}
                for order_id, order in reported.items():
                    state = known[order_id]
                    if (
                        str(state.instrument_id) != order.instrument_id
                        or state.side.value != order.side.value
                        or state.order_quantity != order.order_quantity
                        or state.filled_quantity != order.cumulative_filled
                        or state.remaining_quantity != order.remaining_quantity
                    ):
                        raise RuntimeError(f"活动订单{order_id}数量或归属不一致，需要人工对账")
                    signed = (order.remaining_quantity if order.side is ExecutionSide.BUY
                              else -order.remaining_quantity)
                    working[state.instrument_id] = working.get(state.instrument_id, Decimal(0)) + signed
                self.position_manager.complete_working_recovery(self.client_id, working)
                self._last_orders_revision = snapshot.revision
                self._orders_recovery_pending = False
                self._record("ACTIVE_ORDERS_RECOVERED", f"revision={snapshot.revision} count={len(reported)}")
        except Exception as error:
            self._orders_recovery_pending = True
            self.position_manager.mark_recovery_required(self.client_id)
            self._record("ACTIVE_ORDERS_RECOVERY_FAILED", type(error).__name__)
            raise

    def reconcile(self) -> None:
        """启动/重连均先核对仓位、活动订单恢复条件和权威资金。"""
        with self._submit_lock:
            if self._orders_recovery_pending:
                raise RuntimeError("活动订单尚未经过柜台权威快照恢复")
            if self._heartbeat_in_flight:
                raise RuntimeError("资金心跳查询尚未完成，不能并发对账")
            self._armed = False
            self._account_state = None
            self._heartbeat_in_flight = False
            epoch = self._connection_epoch
        try:
            # 柜台查询可能等待自己的回报回调，不应持有提交/回报锁等待网络。
            super().reconcile()
            state = self.backend.reconcile_account_state()
            with self._submit_lock:
                if epoch != self._connection_epoch or self.report_errors:
                    raise RuntimeError("对账期间发生断线或执行回报冲突")
                if state.account_id != self.account_id:
                    raise ValueError("权威资金快照account_id与执行账户不匹配")
                if state.revision <= self._last_account_revision:
                    raise ValueError("权威资金快照版本未单调递增")
                self._last_account_revision = state.revision
                self._account_state = state
                self._record("RECONCILED", f"position_and_account_revision={state.revision}")
        except Exception as error:
            self.mark_disconnected(type(error).__name__)
            with self._submit_lock:
                self._record("RECONCILE_FAILED", type(error).__name__)
            raise

    def mark_disconnected(self, reason: str) -> None:
        """供连接监控调用；断线立即闭闸，不能依靠旧快照自动重连。"""
        with self._submit_lock:
            self._connection_epoch += 1
            self._armed = False
            self._reconciled = False
            self._account_state = None
            self._heartbeat_in_flight = False
            self._orders_recovery_pending = True
            # 柜台断线后不能假定旧活动订单已经撤销。必须先查询活动订单，
            # 经recover_active_orders校验后替换在途量，再重新查询仓位和资金。
            self.position_manager.mark_recovery_required(self.client_id)
            self._record("DISCONNECTED", reason)

    def refresh_account_state(self) -> AccountStateEvent:
        """周期性只读心跳；查询失败即撤销授权并要求完整重新对账。"""
        with self._submit_lock:
            if not self._started or not self._reconciled:
                raise RuntimeError("账户尚未完成权威对账")
            if self._heartbeat_in_flight:
                raise RuntimeError("已有权威资金查询正在进行")
            self._heartbeat_in_flight = True
            epoch = self._connection_epoch
        try:
            state = self.backend.reconcile_account_state()
            with self._submit_lock:
                if epoch != self._connection_epoch or self.report_errors:
                    raise RuntimeError("资金查询期间发生断线或回报冲突")
                if state.account_id != self.account_id or state.revision <= self._last_account_revision:
                    raise ValueError("权威资金快照账户或版本不匹配")
                self._last_account_revision = state.revision
                self._account_state = state
                self._heartbeat_in_flight = False
                self._record("ACCOUNT_REFRESHED", str(state.revision))
                return state
        except Exception as error:
            self.mark_disconnected(type(error).__name__)
            raise

    def arm_demo(self, confirmation: str) -> None:
        with self._submit_lock:
            if confirmation != self.DEMO_CONFIRMATION:
                raise PermissionError("DEMO下单必须显式确认")
            if not self._started or not self.is_reconciled or self._account_state is None:
                raise RuntimeError("未完成仓位与资金权威对账，不能授权DEMO下单")
            if self._heartbeat_in_flight:
                raise RuntimeError("资金查询期间不能授权DEMO下单")
            if self.report_errors:
                raise RuntimeError("执行回报存在冲突，不能授权DEMO下单")
            if not self._demo_environment_check():
                raise PermissionError("交易节点未被核实为DEMO环境")
            self._armed = True
            self._record("DEMO_ARMED")

    def disarm(self, reason: str = "operator") -> None:
        with self._submit_lock:
            self._armed = False
            self._record("DISARMED", reason)

    def submit_targets(self, request: ExecutionRequest) -> None:
        with self._submit_lock:
            try:
                demo_verified = self._demo_environment_check()
            except Exception:
                demo_verified = False
            if not self._armed or not demo_verified or self._heartbeat_in_flight:
                self._armed = False
                self._record("ORDER_BLOCKED", "not_armed_or_not_demo_or_query_in_flight")
                raise PermissionError("P3客户端处于只读模式或DEMO环境未核实")
            try:
                super().submit_targets(request)
            except Exception as error:
                self._record("ORDER_REJECTED", type(error).__name__)
                raise
            self._record("TARGET_SUBMITTED", request.strategy_id)

    def _apply_report(self, report: ExecutionReport) -> None:
        before = len(self.report_errors)
        super()._apply_report(report)
        if len(self.report_errors) > before:
            self.disarm("report_conflict")
        self._record("REPORT", report.report_type.value)

    def set_risk_mode(
        self,
        mode: KillSwitchMode | str,
        *,
        cancel_active_orders: bool = True,
    ) -> None:
        with self._submit_lock:
            normalized = KillSwitchMode(mode)
            if normalized is not KillSwitchMode.NORMAL:
                self.disarm(f"risk_mode={normalized.value}")
            super().set_risk_mode(normalized, cancel_active_orders=cancel_active_orders)

    def stop(self) -> None:
        with self._submit_lock:
            self._connection_epoch += 1
            self.disarm("stop")
            self._account_state = None
            self._heartbeat_in_flight = False
            working = self.position_manager.snapshot().working_quantities
            if self._started and (
                any(not state.status.is_terminal for state in self.order_state_machine.states())
                or any(key.client_id == self.client_id and quantity != 0
                       for key, quantity in working.items())
            ):
                self._orders_recovery_pending = True
                self.position_manager.mark_recovery_required(self.client_id)
        # 不持有提交锁等待TradingNode线程结束；其最后回报可能需要同一把锁。
        super().stop()
