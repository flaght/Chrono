"""物理账户执行服务；共享连接、目标净额及归属，不运行策略算法。"""

from dataclasses import replace
from decimal import Decimal
import threading

from bomber.framework.trader.portfolio import AccountTargetKey, PortfolioCoordinator, PortfolioCoordinatorState


class AccountExecutionService:
    """一账户一个执行客户端，多策略使用同一服务。

    由装配层持有账户Runtime/锁和完整账户检查点；策略分发端仅持有服务租约。
    此类是进程内服务端口，不创建第二条柜台连接，不自动运行策略。
    """
    def __init__(self, client, *, checkpoint=None):
        self.client = client
        self.client_id = client.client_id
        self.account_id = client.account_id
        self.position_manager = client.position_manager
        self.coordinator = PortfolioCoordinator()
        self._checkpoint = checkpoint
        self._lock = threading.RLock()
        # 发送串行化，但异步回报保存快照不能等待发送结束。
        self._dispatch_lock = threading.RLock()
        self._leases = 0
        self._sources = {}
        self._failed = False
        self._pending = False

    def start(self):
        with self._dispatch_lock:
            with self._lock:
                connect = not self._leases
            if connect:
                self.client.start()
            with self._lock:
                self._leases += 1

    def stop(self):
        with self._dispatch_lock:
            with self._lock:
                if not self._leases:
                    return
                self._leases -= 1
                disconnect = not self._leases
            if disconnect:
                self.client.stop()

    def submit_targets(self, request):
        with self._dispatch_lock:
            self._submit_targets(request)

    def _submit_targets(self, request):
        with self._lock:
            if not self._leases or self._failed or request.client_id != self.client_id:
                raise RuntimeError("账户服务未启动、发送未决或路由不符")
            rows = tuple(dict(row) for row in request.metadata.get("position_attribution", ()))
            if not rows or any(row["strategy_id"] != request.strategy_id for row in rows):
                raise ValueError("账户服务请求须声明单个源策略的完整目标归属")
            targets = {AccountTargetKey(self.client_id, i): Decimal(str(q)) for i, q in request.targets.items()}
            totals = {}
            owners = set()
            for row in rows:
                instrument = row["instrument_id"]
                owner = (row["strategy_id"], row["target_key"], instrument)
                quantity = Decimal(str(row["quantity"]))
                if owner in owners or not quantity.is_finite():
                    raise ValueError("账户服务归属须唯一且数量有限")
                owners.add(owner)
                totals[instrument] = totals.get(instrument, Decimal(0)) + quantity
            if totals != {str(k.instrument_id): q for k, q in targets.items()}:
                raise ValueError("账户服务目标与策略归属不一致")
            previous = self._sources.get(request.strategy_id)
            if previous is not None:
                if request.revision < previous["revision"] or request.ts_event < previous["ts_event"]:
                    raise ValueError("账户服务拒绝旧目标或时间回退")
                if request.revision == previous["revision"] and (targets != previous["targets"] or rows != previous["rows"]):
                    raise ValueError("同版本策略目标不能改变")
            self._sources[request.strategy_id] = dict(revision=request.revision, ts_event=request.ts_event,
                targets=targets, rows=rows)
            revision = self.coordinator.state().strategy_revisions.get(request.strategy_id, 0) + 1
            aggregate = self.coordinator.update(strategy_id=request.strategy_id, revision=revision,
                ts_event=request.ts_event, targets=targets)
            metadata = {**request.metadata, "position_attribution": tuple(row
                for source in self._sources.values() for row in source["rows"])}
            net = replace(request, revision=aggregate.revision,
                targets={key.instrument_id: q for key, q in aggregate.targets.items()},
                logical_targets={}, metadata=metadata)
            self._pending = True
        try:
            if self._checkpoint is not None:
                self._checkpoint()
            self.client.submit_targets(net)
            if self.client.report_errors or not self.client.is_reconciled:
                raise RuntimeError("账户执行回报失败，保留未决发送并关闭账户服务")
            with self._lock:
                self._pending = False
            if self._checkpoint is not None:
                self._checkpoint()
        except Exception:
            # 部分发送/回报无法由一个同步异常推断，整个账户保持闭闸。
            with self._lock:
                self._failed = True
            raise

    def cancel_strategy(self, strategy_id):
        self.client.cancel_strategy(strategy_id)

    def snapshot_state(self):
        with self._lock:
            return {"version": 1, "client_id": self.client_id, "account_id": self.account_id,
                    "failed": self._failed, "pending": self._pending,
                    "portfolio_revision": self.coordinator.state().revision,
                    "strategy_revisions": dict(self.coordinator.state().strategy_revisions),
                    "known_instruments": sorted(str(k.instrument_id) for k in self.coordinator.state().known_keys),
                    "sources": {sid: {"revision": source["revision"], "ts_event": source["ts_event"],
                        "targets": {str(key.instrument_id): str(q) for key, q in source["targets"].items()},
                        "rows": [dict(row) for row in source["rows"]]} for sid, source in self._sources.items()}}

    def restore_state(self, state):
        from bomber.framework.market.basic.base import InstrumentId
        with self._dispatch_lock, self._lock:
            if self._leases or state.get("version") != 1 or state.get("client_id") != self.client_id or state.get("account_id") != self.account_id:
                raise ValueError("账户服务恢复身份不符或已启动")
            sources, coordinator = {}, PortfolioCoordinator()
            for sid, source in state["sources"].items():
                targets = {AccountTargetKey(self.client_id, InstrumentId.from_str(i)): Decimal(q)
                           for i, q in source["targets"].items()}
                rows = tuple(dict(row) for row in source["rows"])
                totals = {}
                owners = set()
                for row in rows:
                    owner = (row["strategy_id"], row["target_key"], row["instrument_id"])
                    if row["strategy_id"] != sid or owner in owners:
                        raise ValueError("账户服务恢复存在其他策略或重复归属")
                    owners.add(owner)
                    totals[row["instrument_id"]] = totals.get(row["instrument_id"], Decimal(0)) + Decimal(row["quantity"])
                if totals != {str(k.instrument_id): q for k, q in targets.items()} or any(not q.is_finite() for q in totals.values()):
                    raise ValueError("账户服务恢复目标与归属不一致")
                coordinator.update(strategy_id=sid, revision=source["revision"], ts_event=source["ts_event"], targets=targets)
                sources[sid] = dict(revision=source["revision"], ts_event=source["ts_event"], targets=targets, rows=rows)
            coordinator.restore(PortfolioCoordinatorState(revision=state["portfolio_revision"],
                contributions={sid: source["targets"] for sid, source in sources.items()},
                strategy_revisions=state["strategy_revisions"],
                known_keys=frozenset(AccountTargetKey(self.client_id, InstrumentId.from_str(i)) for i in state["known_instruments"])))
            self._sources, self.coordinator = sources, coordinator
            # 恢复后的账户工作订单仍须由客户端执行自己的恢复协议。
            self._pending = bool(state["pending"])
            self._failed = bool(state["failed"]) or self._pending
