"""一个策略执行账本向多个账户分发目标；账户成交不回写主策略账本。"""

from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal
import threading


@dataclass(frozen=True)
class AccountExecutionRoute:
    client: object
    multiplier: Decimal = Decimal(1)

    def __post_init__(self):
        value = Decimal(str(self.multiplier))
        if not value.is_finite() or value <= 0:
            raise ValueError("账户目标倍率须为有限正数")
        object.__setattr__(self, "multiplier", value)


class TargetFanoutClient:
    """执行端口组合；position_client是唯一策略仓位及策略回调来源。

    followers可以是原生/第三方执行客户端，也可以是账户执行服务租约。
    各端独立规划、风控及记账；账户失败隔离并记录，未决发送须显式处理。
    checkpoint应连接状态管理器，在每次发送前后写入分发日志。
    """
    def __init__(self, position_client, followers=(), *, checkpoint=None):
        self.position_client = position_client
        self.client_id = position_client.client_id
        self.position_manager = position_client.position_manager
        self.routes = tuple(followers)
        ids = [r.client.client_id for r in self.routes]
        accounts = [getattr(r.client, "account_id", r.client.client_id) for r in self.routes]
        if (len(ids) != len(set(ids)) or len(accounts) != len(set(accounts)) or self.client_id in ids
                or getattr(position_client, "account_id", self.client_id) in accounts):
            raise ValueError("分发账户/客户端须唯一，不能重复发送至策略仓位来源端")
        if any(getattr(r.client, "position_manager", None) is self.position_manager for r in self.routes):
            raise ValueError("跟随账户不能共用主策略仓位账本")
        self._checkpoint = checkpoint
        self._started = False
        self._lock = threading.RLock()
        self._dispatch_lock = threading.RLock()
        self._journal = {}
        self._blocked = set()
        self._started_clients = []

    def register_execution_event_handler(self, handler):
        self.position_client.register_execution_event_handler(handler)

    @property
    def is_reconciled(self):
        return self.position_client.is_reconciled

    @property
    def backend(self):
        return self.position_client.backend

    @property
    def order_state_machine(self):
        return self.position_client.order_state_machine

    @property
    def report_errors(self):
        return self.position_client.report_errors

    @property
    def account_reconciliation(self):
        return self.position_client.account_reconciliation

    @property
    def account_observation(self):
        return self.position_client.account_observation

    def disarm(self, reason="operator"):
        self.position_client.disarm(reason)

    def start(self):
        with self._dispatch_lock:
            with self._lock:
                if self._started:
                    return
            started = []
            try:
                self.position_client.start()
                started.append(self.position_client)
                for route in self.routes:
                    client = route.client
                    try:
                        client.start()
                    except Exception as error:
                        with self._lock:
                            self._blocked.add(client.client_id)
                            self._journal[client.client_id] = {"status": "blocked", "error": str(error)}
                        self._save()
                    else:
                        started.append(client)
            except BaseException:
                for client in reversed(started):
                    client.stop()
                raise
            with self._lock:
                self._started = True
                self._started_clients = started

    def stop(self):
        with self._dispatch_lock:
            with self._lock:
                if not self._started:
                    return
            errors = []
            for client in reversed(self._started_clients):
                try:
                    client.stop()
                except Exception as error:
                    errors.append(error)
            with self._lock:
                self._started = False
                self._started_clients = []
            if errors:
                raise RuntimeError("分发端停止失败") from errors[0]

    def _save(self):
        if self._checkpoint is not None:
            self._checkpoint()

    def set_checkpoint(self, checkpoint):
        with self._dispatch_lock, self._lock:
            if self._started or self._checkpoint is not None:
                raise RuntimeError("分发写前保存只能在启动前绑定一次")
            self._checkpoint = checkpoint

    def register_submit_failure_handler(self, handler):
        self.position_client.register_submit_failure_handler(handler)

    def submit_targets(self, request):
        with self._dispatch_lock:
            with self._lock:
                if not self._started or request.client_id != self.client_id:
                    raise RuntimeError("分发端未启动或请求客户端不匹配")
            # 策略执行独立推进；跟随账户不能污染仓位或阻止策略信号模型。
            self.position_client.submit_targets(request)
            if self.position_client.report_errors or not self.position_client.is_reconciled:
                raise RuntimeError("策略仓位来源回报失败，停止向跟随账户分发")
            for route in self.routes:
                client = route.client
                with self._lock:
                    if client.client_id in self._blocked:
                        continue
                metadata = dict(request.metadata)
                if "position_attribution" in metadata:
                    metadata["position_attribution"] = tuple({**row,
                        "quantity": str(Decimal(str(row["quantity"])) * route.multiplier)}
                        for row in metadata["position_attribution"])
                metadata["origin_client_id"] = self.client_id
                child = replace(request, client_id=client.client_id,
                    targets={i: Decimal(str(q)) * route.multiplier for i, q in request.targets.items()},
                    logical_targets={key: Decimal(str(q)) * route.multiplier for key, q in request.logical_targets.items()},
                    metadata=metadata)
                entry = {"strategy_id": request.strategy_id, "revision": request.revision,
                         "ts_event": request.ts_event, "status": "pending"}
                with self._lock:
                    self._journal[client.client_id] = entry
                try:
                    self._save()  # 写前失败不得到达第三方交易端。
                    client.submit_targets(child)
                    with self._lock:
                        entry["status"] = "submitted"  # 受理不等于成交。
                    self._save()
                except Exception as error:
                    with self._lock:
                        entry.update(status="blocked", error=str(error))
                        self._blocked.add(client.client_id)
                    self._save()

    def cancel_strategy(self, strategy_id):
        errors = []
        for client in (self.position_client, *(r.client for r in self.routes)):
            try:
                client.cancel_strategy(strategy_id)
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError("分发账户撤单未全部完成") from errors[0]

    @property
    def dispatch_status(self):
        with self._lock:
            return deepcopy(self._journal)

    def snapshot_state(self):
        with self._lock:
            return {"version": 1, "client_id": self.client_id,
                    "position_account_id": getattr(self.position_client, "account_id", self.client_id),
                    "routes": [[r.client.client_id, getattr(r.client, "account_id", r.client.client_id), str(r.multiplier)] for r in self.routes],
                    "journal": deepcopy(self._journal), "blocked": sorted(self._blocked)}

    def restore_state(self, state):
        with self._dispatch_lock, self._lock:
            current = self.snapshot_state()
            if self._started or any(state.get(key) != current[key] for key in ("version", "client_id", "position_account_id", "routes")):
                raise ValueError("分发恢复须在启动前且账户、倍率及策略来源不变")
            journal = deepcopy(state.get("journal", {}))
            allowed = {r.client.client_id for r in self.routes}
            blocked = set(state.get("blocked", ()))
            if not journal.keys() <= allowed or not blocked <= allowed:
                raise ValueError("分发恢复包含未知账户")
            for client_id, entry in journal.items():
                if entry.get("status") not in {"pending", "submitted", "blocked"}:
                    raise ValueError("分发状态无效")
                if entry["status"] in {"pending", "blocked"}:
                    blocked.add(client_id)
            self._journal, self._blocked = journal, blocked
