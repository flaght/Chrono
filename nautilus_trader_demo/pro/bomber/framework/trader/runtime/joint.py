"""GAP-10：固定路由联合会话，DIRECT非原子执行及失败后整体闭闸。"""

from contextlib import ExitStack
from dataclasses import dataclass
import threading
import time

from bomber.framework.trader.contracts import ExecutionRequest, ExecutionRoute, RuntimeMode
from bomber.framework.trader.execution.events import OrderEventStatus, OrderUpdateEvent
from bomber.framework.trader.runner import UnifiedStrategyRunner
from .ownership import account_writer_lease
from bomber.framework.market.basic.base import InstrumentId


@dataclass(frozen=True)
class JointExecutionBinding:
    execution: object
    identity: object
    ready: object


class _JointBackend:
    def __init__(self, runner, backend):
        self.runner, self.backend = runner, backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def submit_order(self, order):
        # Planner可能拆出多笔订单；每笔实际发送前重新检查共同闸门。
        self.runner.require_ready()
        with self.runner._joint_lock:
            count = self.runner._order_attempts.get(order.backend_id, 0)
            if self.runner.max_orders_per_client is not None and count >= self.runner.max_orders_per_client:
                self.runner.trip(f"{order.backend_id}:session_order_cap")
                raise RuntimeError("联合会话已达该端报单上限")
            self.runner._order_attempts[order.backend_id] = count + 1
        return self.backend.submit_order(order)


class _JointClient:
    """保留原客户端事件与状态机，只在每端发送前再查共同闸门。"""
    def __init__(self, runner, binding):
        self.runner = runner
        self.binding = binding
        self.client = binding.execution.client
        self.client_id = self.client.client_id
        self.client.register_execution_event_handler(self._observe)

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _observe(self, event):
        if isinstance(event, OrderUpdateEvent) and event.status is OrderEventStatus.REJECTED:
            self.runner.trip(f"{self.client_id}:order_rejected")

    def submit_targets(self, request):
        self.runner.require_ready()
        row = {"client_id": self.client_id, "strategy_id": request.strategy_id,
            "revision": request.revision, "ts_event": request.ts_event,
            "targets": {str(k): str(v) for k, v in request.targets.items()}, "status": "submitting"}
        self.runner.audit.append(row)
        try:
            self.client.submit_targets(request)
            # 可能仅规划0笔订单；submitted表示请求已处理，不表示已成交。
            row["status"] = "submitted"
        except Exception as error:
            row["status"] = "failed"
            row["error_type"] = type(error).__name__
            self.runner.trip(f"{self.client_id}:submission_failed")
            raise


class JointLiveRunner(UnifiedStrategyRunner):
    """所有必需端就绪才发目标；失败锁存，恢复不自动重放。

    原客户端继续维护独立订单状态机、账户和归属；共享PositionManager。
    拒单、断线、查询失败或发送异常关闭整个组合。首期HALTED，既有仓位
    保留，不自动平仓/补偿；撤单由控制线程显式执行，回调内不等网络。
    """

    def __init__(self, *, position_manager, bindings, lease_factory=account_writer_lease,
                 account_max_age_ns=30_000_000_000, clock_ns=time.time_ns, max_orders_per_client=None):
        super().__init__(RuntimeMode.LIVE, position_manager=position_manager)
        self.bindings = tuple(bindings)
        clients = [item.execution.client.client_id for item in self.bindings]
        if len(clients) < 2 or len(set(clients)) != len(clients):
            raise ValueError("联合会话须有至少两个不同客户端")
        if len({item.identity.key for item in self.bindings}) != len(self.bindings):
            raise ValueError("同一物理账户不能绑定多个活动写入客户端")
        if account_max_age_ns <= 0:
            raise ValueError("账户快照时效须为正数")
        if max_orders_per_client is not None and max_orders_per_client < 1:
            raise ValueError("单端会话报单上限须为正数")
        self.max_orders_per_client = max_orders_per_client
        self._order_attempts = {}
        self.account_max_age_ns = account_max_age_ns
        self.clock_ns = clock_ns
        self._joint_lock = threading.RLock()
        self._lease_factory = lease_factory
        self._resources = ExitStack()
        self._lease_held = False
        self._closed = False
        self._failure = None
        self._enabled = False
        self.audit = []
        self._orders_revisions = {}
        for item in self.bindings:
            client = item.execution.client
            if item.execution.positions is not position_manager:
                raise ValueError("联合执行须共享PositionManager")
            for name in ("arm_demo", "disarm", "recover_active_orders", "refresh_account_state"):
                if not callable(getattr(client, name, None)):
                    raise ValueError("联合会话只接受受控在线客户端")
        for item in self.bindings:
            client = item.execution.client
            client.backend = _JointBackend(self, client.backend)
            self.add_market_observer(item.execution.prices)
            self.add_execution_client(_JointClient(self, item))
        self._joint_clients_installed = True

    def add_execution_client(self, client):
        if getattr(self, "_joint_clients_installed", False):
            raise RuntimeError("联合客户端须在构造时绑定物理账户，不能另添未受保护客户端")
        super().add_execution_client(client)

    @property
    def failure(self):
        with self._joint_lock:
            return self._failure

    def trip(self, reason):
        # 柜台回调内只锁存，避免跨客户端锁与网络线程相互等待。
        with self._joint_lock:
            self._enabled = False
            if self._failure is None:
                self._failure = str(reason)
                self.audit.append({"action": "HALTED", "reason": self._failure})

    def readiness(self, *, require_armed=True):
        result = {}
        for item in self.bindings:
            client = item.execution.client
            state = client.account_state
            age = None if state is None else self.clock_ns() - state.ts_event
            try:
                connected = bool(item.ready())
            except Exception:
                connected = False
            result[client.client_id] = bool(self._lease_held and connected
                and client.is_reconciled and state is not None
                and age is not None and 0 <= age <= self.account_max_age_ns
                and not client.report_errors
                and not getattr(client, "_heartbeat_in_flight", False)
                and not self.position_manager.is_recovery_required(client.client_id)
                and client.client_id in self._orders_revisions
                and (not require_armed or client.is_armed))
        return result

    def require_ready(self):
        if not self._enabled or self.failure or not all(self.readiness().values()):
            if self._enabled:
                self.trip("required_endpoint_not_ready")
            raise RuntimeError("联合会话未就绪或已闭闸；须双端权威对账及显式授权")

    def start(self):
        if self._closed:
            raise RuntimeError("已结束的联合会话不能重新启动")
        if self._started:
            return
        try:
            self._resources.enter_context(self._lease_factory(tuple(item.identity for item in self.bindings)))
            self._lease_held = True
            super().start()
            self.refresh_authority()
        except BaseException:
            self.trip("startup_failed")
            self.stop()
            raise

    def refresh_authority(self):
        """资金、全账户仓位和全量活动订单；查询变化/冲突保持闭闸。"""
        self._enabled = False
        for item in self.bindings:
            item.execution.client.disarm("joint_authority_check")
        try:
            for item in self.bindings:
                client = item.execution.client
                if self.position_manager.is_recovery_required(client.client_id):
                    client.recover_active_orders()
                client.reconcile()
                snapshot = client.backend.reconcile_active_orders()
                if (snapshot.client_id != client.client_id or snapshot.account_id != client.account_id
                        or snapshot.revision <= self._orders_revisions.get(client.client_id, 0)):
                    raise RuntimeError("活动订单身份或版本冲突")
                known = {state.client_order_id: state for state in client.order_state_machine.states()
                    if not state.status.is_terminal}
                reported = {order.client_order_id: order for order in snapshot.orders}
                if known.keys() != reported.keys():
                    raise RuntimeError("活动订单不一致；查清原订单前禁止恢复")
                for key, order in reported.items():
                    state = known[key]
                    if (str(state.instrument_id), state.side.value, state.order_quantity,
                            state.filled_quantity, state.remaining_quantity) != (
                            order.instrument_id, order.side.value, order.order_quantity,
                            order.cumulative_filled, order.remaining_quantity):
                        raise RuntimeError("活动订单数量或方向不一致")
                instruments = {key.instrument_id for key in self.position_manager.snapshot().account_positions
                    if key.client_id == client.client_id}
                instruments.update(InstrumentId.from_str(row["instrument_id"]) for row in
                    self.position_manager.attribution.state()["owned"] if row["client_id"] == client.client_id)
                for instrument in instruments:
                    if self.position_manager.unassigned_position(client.client_id, instrument) != 0:
                        raise RuntimeError("权威仓位存在未归属差额，不能联合恢复")
                if not client.is_reconciled or self.position_manager.is_recovery_required(client.client_id):
                    raise RuntimeError("权威查询期间发生断线，不能解除联合闸门")
                self._orders_revisions[client.client_id] = snapshot.revision
            self.audit.append({"action": "AUTHORITY_CHECKED"})
        except Exception:
            self.trip("authority_check_failed")
            raise

    def arm(self, confirmation):
        if confirmation != "AUTHORIZE_JOINT_DEMO_ORDERS":
            raise PermissionError("联合SimNow/DEMO报单须显式确认")
        if not self._started or self.failure or not all(self.readiness(require_armed=False).values()):
            raise RuntimeError("双端尚未就绪，不能授权")
        try:
            for item in self.bindings:
                item.execution.client.arm_demo(item.execution.client.DEMO_CONFIRMATION)
            self._enabled = True
            self.require_ready()
            self.audit.append({"action": "JOINT_ARMED"})
        except Exception:
            self.trip("authorization_failed")
            for item in self.bindings:
                item.execution.client.disarm("joint_authorization_failed")
            raise

    def recover(self, *, operator, reason):
        if not operator.strip() or not reason.strip() or not self.failure or not self._started:
            raise ValueError("恢复须为运行中的闭闸会话，并记录操作人及原因")
        self.refresh_authority()
        if not all(self.readiness(require_armed=False).values()):
            raise RuntimeError("必需端尚未就绪")
        self.audit.append({"action": "RECOVERED_DISARMED", "operator": operator, "reason": reason})
        with self._joint_lock:
            self._failure = None
        # 不授权、不自动重放、不切换原客户端。arm和retry分别显式调用。

    def _submit_locked(self, intent):
        self.require_ready()  # 早于TargetStore，未就绪目标不占revision。
        if intent.execution_policy != "DIRECT":
            raise ValueError("联合首期只支持非原子DIRECT，不支持自动补偿政策")
        if intent.deadline_ns is not None and self.clock_ns() > intent.deadline_ns:
            raise RuntimeError("联合目标已过期")
        try:
            super()._submit_locked(intent)
        except Exception:
            self.trip("target_dispatch_failed")
            raise

    def retry_remaining(self, strategy_id, *, now_ns):
        """对账/再授权后按保留目标与当前仓位重新规划；不直接重放旧订单。"""
        with self._submit_lock:
            self.require_ready()
            intent = self.target_store.get(strategy_id)
            if intent is None or (intent.deadline_ns is not None and now_ns > intent.deadline_ns):
                raise RuntimeError("保留目标不存在或已过期")
            self.market_health_gate.check_target(intent, intent)
            if self.market_health_gate.snapshot(strategy_id).access_mode.value != "NORMAL":
                raise RuntimeError("联合剩余目标须先完成行情恢复确认")
            registration = self._registrations[strategy_id]
            snapshot = self.portfolio_coordinator.snapshot(ts_event=now_ns)
            for client_id in sorted({route.client_id for route in registration.execution_routes.values()}):
                self._clients[client_id].submit_targets(ExecutionRequest(strategy_id, intent.revision,
                    client_id, now_ns, {key.instrument_id: qty for key, qty in snapshot.targets.items()
                        if key.client_id == client_id}, intent.execution_policy, intent.deadline_ns,
                    logical_targets={key: intent.targets.get(key, 0) for key, route in
                        registration.execution_routes.items() if route.client_id == client_id},
                    metadata={**intent.metadata, "signal_ts_event": intent.ts_event,
                        "joint_retry": True, "position_attribution": self._position_attribution_targets(
                            snapshot, client_id)}))

    def cancel_and_confirm(self, *, timeout_seconds=15):
        if timeout_seconds <= 0:
            raise ValueError("撤单等待须为正数")
        self._enabled = False
        for item in self.bindings:
            client = item.execution.client
            client.disarm("joint_cancel")
            for strategy_id in self._registrations:
                client.cancel_strategy(strategy_id)
        deadline = time.monotonic() + timeout_seconds
        while True:
            snapshots = [item.execution.client.backend.reconcile_active_orders() for item in self.bindings]
            if all(not snapshot.orders for snapshot in snapshots):
                self.refresh_authority()
                return
            if time.monotonic() >= deadline:
                self.trip("cancel_confirmation_timeout")
                raise RuntimeError("双端撤单未完成，保留原订单及仓位证据")
            time.sleep(0.2)

    def stop(self):
        if self._closed:
            return
        self.trip("stop")
        errors = []
        # 启动失败可能只启动了部分端；每个已创建积木仍须尝试释放。
        for feed in reversed(tuple(self._feeds.values())):
            try:
                feed.disconnect()
            except Exception as error:
                errors.append(error)
        for registration in reversed(tuple(self._registrations.values())):
            try:
                registration.strategy._stop()
                registration.strategy._unbind()
            except Exception as error:
                errors.append(error)
        for item in reversed(self.bindings):
            try:
                item.execution.client.disarm("joint_stop")
                item.execution.client.stop()
            except Exception as error:
                errors.append(error)
            try:
                if item.execution.backend is not None:
                    item.execution.backend.stop()
                if item.execution.driver is not None:
                    item.execution.driver.stop()
            except Exception as error:
                errors.append(error)
        self._started = False
        if errors:
            # 不释放账户租约；调用方处理停机失败后可以重试stop。
            raise RuntimeError("联合停机未完成，账户锁保留: " + "; ".join(str(e) for e in errors))
        self._resources.close()
        self._lease_held = False
        self._closed = True


def assemble_joint_strategy(runner, strategy, *, feeds, bindings):
    routes = tuple(bindings.execution)
    if bindings.role_guard is not None or any(not isinstance(route, ExecutionRoute) for route in routes):
        raise ValueError("联合首期只支持固定真实合约路由")
    if {route.client_id for route in routes} != {item.execution.client.client_id for item in runner.bindings}:
        raise ValueError("联合策略须声明全部必需交易端的路由")
    for feed_id, feed in feeds.items():
        runner.add_data_feed(feed_id, feed)
    runner.add_strategy(strategy, data_bindings=bindings.data, execution_routes=routes,
        time_feed_ids=bindings.time_feed_ids)
    return runner
