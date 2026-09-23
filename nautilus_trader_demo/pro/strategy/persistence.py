"""策略执行核心状态的版本化、校验和保护及原子JSON持久化。"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategy.execution.live.client import BackendExecutionClient

from market.basic.base import InstrumentId
from strategy.contracts import TargetPortfolio, TargetUpdateMode
from strategy.execution.contracts import OrderIntent, OrderSide
from strategy.execution.ctp.native_driver import (
    CtpDriverCheckpoint,
    CtpNativeTraderDriver,
    CtpOrderAssociation,
)
from strategy.execution.ctp.ledger import (
    CtpLedgerState,
    CtpPositionLedger,
    CtpPositionSnapshot,
)
from strategy.execution.order_state import (
    OrderLifecycleStatus,
    OrderReportStateMachine,
    OrderState,
    OrderStateCheckpoint,
)
from strategy.portfolio import (
    AccountReconciliationState,
    AccountTargetKey,
    PortfolioCoordinator,
    PortfolioCoordinatorState,
    PositionManager,
    PositionManagerState,
    PositionSnapshot,
    TargetStore,
)


SCHEMA_VERSION = 1


class StatePersistenceError(RuntimeError):
    pass


class StateCorruptionError(StatePersistenceError):
    pass


class ConcurrentStateWriteError(StatePersistenceError):
    pass


@dataclass(frozen=True)
class PersistedState:
    generation: int
    saved_at_ns: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class RecoverySummary:
    generation: int
    saved_at_ns: int
    targets: int
    orders: int
    ctp_accounts: int
    requires_authoritative_reconciliation: bool = True


class JsonStateStore:
    """单文件状态存储，使用CAS generation、防损坏校验和与原子替换。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def load(self) -> PersistedState | None:
        with self._lock:
            if not self.path.exists():
                return None
            try:
                document = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StateCorruptionError(f"无法读取状态文件: {self.path}") from exc
            checksum = document.pop("checksum", None)
            if not isinstance(checksum, str) or checksum != _checksum(document):
                raise StateCorruptionError("状态文件校验和不匹配")
            if document.get("schema_version") != SCHEMA_VERSION:
                raise StatePersistenceError(
                    f"不支持的状态schema_version: {document.get('schema_version')}",
                )
            generation = document.get("generation")
            saved_at_ns = document.get("saved_at_ns")
            payload = document.get("payload")
            if not isinstance(generation, int) or generation < 1:
                raise StateCorruptionError("状态文件generation无效")
            if not isinstance(saved_at_ns, int) or saved_at_ns < 0:
                raise StateCorruptionError("状态文件saved_at_ns无效")
            if not isinstance(payload, dict):
                raise StateCorruptionError("状态文件payload无效")
            return PersistedState(generation, saved_at_ns, payload)

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        expected_generation: int,
    ) -> PersistedState:
        with self._lock:
            current = self.load()
            current_generation = 0 if current is None else current.generation
            if current_generation != expected_generation:
                raise ConcurrentStateWriteError(
                    f"状态版本已变化: expected={expected_generation} "
                    f"current={current_generation}",
                )
            generation = current_generation + 1
            saved_at_ns = time.time_ns()
            document = {
                "schema_version": SCHEMA_VERSION,
                "generation": generation,
                "saved_at_ns": saved_at_ns,
                "payload": dict(payload),
            }
            document["checksum"] = _checksum(document)
            encoded = json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary_path = Path(handle.name)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary_path, 0o600)
                os.replace(temporary_path, self.path)
                temporary_path = None
                _fsync_directory(self.path.parent)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            return PersistedState(generation, saved_at_ns, dict(payload))


class RuntimeStateManager:
    """统一保存/恢复Target、组合、仓位、订单状态和CTP账本。"""

    def __init__(
        self,
        repository: JsonStateStore,
        target_store: TargetStore,
        portfolio_coordinator: PortfolioCoordinator,
        position_manager: PositionManager,
        *,
        order_machines: Mapping[str, OrderReportStateMachine] | None = None,
        ctp_ledgers: Mapping[str, CtpPositionLedger] | None = None,
        ctp_drivers: Mapping[str, CtpNativeTraderDriver] | None = None,
    ) -> None:
        self.repository = repository
        self.target_store = target_store
        self.portfolio_coordinator = portfolio_coordinator
        self.position_manager = position_manager
        self.order_machines = dict(order_machines or {})
        self.ctp_ledgers = dict(ctp_ledgers or {})
        self.ctp_drivers = dict(ctp_drivers or {})
        self._ctp_submit_locks: dict[str, Any] = {}
        self._generation = 0
        self._lock = threading.RLock()

    @property
    def generation(self) -> int:
        return self._generation

    def enable_ctp_autosave(
        self, client_id: str, client: BackendExecutionClient,
    ) -> None:
        """报单写前落盘、柜台回报后落盘；失败时由Driver关闭闸门。"""
        driver = self.ctp_drivers.get(client_id)
        machine = self.order_machines.get(client_id)
        if (
            driver is None or machine is None
            or client.client_id != client_id
            or client.order_state_machine is not machine
            or client.position_manager is not self.position_manager
            or getattr(client.backend, "driver", None) is not driver
        ):
            raise ValueError("CTP自动保存要求同一客户端、订单状态机、仓位与Driver")

        def before_send(order_id: str, intent: OrderIntent) -> None:
            with client._submit_lock:
                machine.register_pending(order_id, intent)
                try:
                    self.save()
                except Exception:
                    machine.discard_pending(order_id)
                    raise

        def on_unsent(order_id: str) -> None:
            with client._submit_lock:
                if machine.state(order_id) is not None:
                    machine.discard_pending(order_id)

        def save_after_transition() -> None:
            with client._submit_lock:
                self.save()

        # 同步发送失败时，客户端先撤回在途量，再保存修正后的检查点。
        client.register_submit_failure_handler(save_after_transition)
        driver.bind_durability(before_send, save_after_transition, on_unsent)
        self._ctp_submit_locks[client_id] = client._submit_lock

    def save(self) -> PersistedState:
        # 所有保存路径按“客户端提交锁 → Manager锁”排序，防止柜台回调与外部保存交叉。
        with ExitStack() as stack:
            for client_id in sorted(self._ctp_submit_locks):
                stack.enter_context(self._ctp_submit_locks[client_id])
            return self._save_with_manager_lock()

    def _save_with_manager_lock(self) -> PersistedState:
        with self._lock:
            driver_checkpoints = {
                client_id: driver.checkpoint()
                for client_id, driver in self.ctp_drivers.items()
            }
            for client_id in self.ctp_drivers:
                if client_id not in self.order_machines:
                    raise StatePersistenceError("CTP Driver缺少同客户端订单状态机")
                checkpoint = driver_checkpoints[client_id]
                local = {
                    item.state.client_order_id: item.state
                    for item in self.order_machines[client_id].checkpoints()
                    if not item.state.status.is_terminal
                }
                associated = {item.client_order_id: item for item in checkpoint.orders}
                if local.keys() != associated.keys() or any(
                    local[key].filled_quantity != item.filled_quantity
                    or local[key].order_quantity != item.intent.quantity
                    or local[key].instrument_id != item.intent.instrument_id
                    or local[key].side != item.intent.side
                    or (local[key].last_sequence or 0) != item.sequence
                    for key, item in associated.items()
                ):
                    raise StatePersistenceError("CTP Driver与订单状态机尚未达到同一状态，拒绝保存")
            payload = {
                "targets": _encode_targets(self.target_store.all()),
                "portfolio": _encode_portfolio(self.portfolio_coordinator.state()),
                "positions": _encode_positions(self.position_manager.state()),
                "orders": {
                    client_id: _encode_orders(machine.checkpoints())
                    for client_id, machine in sorted(self.order_machines.items())
                },
                "ctp_ledgers": {
                    account_id: _encode_ctp_ledger(ledger.state())
                    for account_id, ledger in sorted(self.ctp_ledgers.items())
                },
                "ctp_drivers": {
                    client_id: _encode_ctp_driver(checkpoint)
                    for client_id, checkpoint in sorted(driver_checkpoints.items())
                },
            }
            persisted = self.repository.save(
                payload,
                expected_generation=self._generation,
            )
            self._generation = persisted.generation
            return persisted

    def restore(self) -> RecoverySummary | None:
        with self._lock:
            persisted = self.repository.load()
            if persisted is None:
                return None
            payload = persisted.payload
            targets = _decode_targets(payload.get("targets"))
            portfolio = _decode_portfolio(payload.get("portfolio"))
            positions = _decode_positions(payload.get("positions"))
            order_payload = payload.get("orders")
            ledger_payload = payload.get("ctp_ledgers")
            driver_payload = payload.get("ctp_drivers", {})
            if not isinstance(order_payload, dict) or set(order_payload) != set(self.order_machines):
                raise StatePersistenceError("订单状态机集合与持久化快照不一致")
            if not isinstance(ledger_payload, dict) or set(ledger_payload) != set(self.ctp_ledgers):
                raise StatePersistenceError("CTP账本集合与持久化快照不一致")
            if not isinstance(driver_payload, dict) or set(driver_payload) != set(self.ctp_drivers):
                raise StatePersistenceError("CTP Driver集合与持久化快照不一致")
            orders = {
                client_id: _decode_orders(order_payload[client_id])
                for client_id in self.order_machines
            }
            ledgers = {
                account_id: _decode_ctp_ledger(ledger_payload[account_id])
                for account_id in self.ctp_ledgers
            }
            drivers = {
                client_id: _decode_ctp_driver(driver_payload[client_id])
                for client_id in self.ctp_drivers
            }
            for client_id, checkpoint in drivers.items():
                if checkpoint.driver_id != client_id or client_id not in orders:
                    raise StatePersistenceError("CTP Driver检查点缺少同客户端订单状态机")
                local = {
                    item.state.client_order_id: item.state
                    for item in orders[client_id] if not item.state.status.is_terminal
                }
                associated = {
                    item.client_order_id: item for item in checkpoint.orders
                }
                if local.keys() != associated.keys() or any(
                    local[key].filled_quantity != item.filled_quantity
                    or local[key].order_quantity != item.intent.quantity
                    or local[key].instrument_id != item.intent.instrument_id
                    or local[key].side != item.intent.side
                    or (local[key].last_sequence or 0) != item.sequence
                    for key, item in associated.items()
                ):
                    raise StatePersistenceError("CTP Driver关联与订单状态机检查点不一致")

            old_targets = self.target_store.all()
            old_portfolio = self.portfolio_coordinator.state()
            old_positions = self.position_manager.state()
            old_orders = {
                key: machine.checkpoints()
                for key, machine in self.order_machines.items()
            }
            old_ledgers = {
                key: ledger.state() for key, ledger in self.ctp_ledgers.items()
            }
            staged_drivers: list[CtpNativeTraderDriver] = []
            try:
                self.target_store.restore(targets)
                self.portfolio_coordinator.restore(portfolio)
                self.position_manager.restore(positions)
                # 本地保存的“已对账”标志不能跨进程继承；账户仓位必须重新查询柜台。
                for client_id in positions.snapshot.account_reconciliations:
                    self.position_manager.clear_account_reconciliation(client_id)
                for key, machine in self.order_machines.items():
                    machine.restore(orders[key])
                for key, ledger in self.ctp_ledgers.items():
                    ledger.restore(ledgers[key])
                recovery_clients = {
                    key.client_id
                    for key, quantity in positions.snapshot.working_quantities.items()
                    if quantity != 0
                }
                recovery_clients.update(
                    client_id
                    for client_id, checkpoints in orders.items()
                    if any(not item.state.status.is_terminal for item in checkpoints)
                )
                # CTP即使本地记录为空，也要查询柜台确认没有未知活动订单。
                recovery_clients.update(self.ctp_drivers)
                for client_id in recovery_clients:
                    self.position_manager.mark_recovery_required(client_id)
                for key, driver in self.ctp_drivers.items():
                    driver.stage_recovery(drivers[key])
                    staged_drivers.append(driver)
            except Exception:
                for driver in staged_drivers:
                    driver.discard_staged_recovery()
                self.target_store.restore(old_targets)
                self.portfolio_coordinator.restore(old_portfolio)
                self.position_manager.restore(old_positions)
                for key, machine in self.order_machines.items():
                    machine.restore(old_orders[key])
                for key, ledger in self.ctp_ledgers.items():
                    ledger.restore(old_ledgers[key])
                raise
            self._generation = persisted.generation
            return RecoverySummary(
                generation=persisted.generation,
                saved_at_ns=persisted.saved_at_ns,
                targets=len(targets),
                orders=sum(len(items) for items in orders.values()),
                ctp_accounts=len(ledgers),
            )


def _checksum(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("metadata的映射键必须是字符串")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"不支持持久化的metadata类型: {type(value).__name__}")


def _restore_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__decimal__"}:
            return Decimal(value["__decimal__"])
        return {key: _restore_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_json_value(item) for item in value]
    return value


def _encode_targets(targets: Mapping[str, TargetPortfolio]) -> list[dict[str, Any]]:
    return [
        {
            "strategy_id": item.strategy_id,
            "revision": item.revision,
            "ts_event": item.ts_event,
            "targets": {key: str(value) for key, value in item.targets.items()},
            "execution_policy": item.execution_policy,
            "deadline_ns": item.deadline_ns,
            "metadata": _json_value(item.metadata),
        }
        for _, item in sorted(targets.items())
    ]


def _decode_targets(value: Any) -> dict[str, TargetPortfolio]:
    if not isinstance(value, list):
        raise StateCorruptionError("targets格式无效")
    result: dict[str, TargetPortfolio] = {}
    for item in value:
        target = TargetPortfolio(
            strategy_id=item["strategy_id"],
            revision=item["revision"],
            ts_event=item["ts_event"],
            targets={key: Decimal(number) for key, number in item["targets"].items()},
            execution_policy=item["execution_policy"],
            deadline_ns=item["deadline_ns"],
            metadata=_restore_json_value(item["metadata"]),
            update_mode=TargetUpdateMode.REPLACE,
        )
        if target.strategy_id in result:
            raise StateCorruptionError("targets包含重复strategy_id")
        result[target.strategy_id] = target
    return result


def _account_key(key: AccountTargetKey) -> dict[str, str]:
    return {"client_id": key.client_id, "instrument_id": str(key.instrument_id)}


def _decode_account_key(value: Mapping[str, str]) -> AccountTargetKey:
    return AccountTargetKey(value["client_id"], InstrumentId.from_str(value["instrument_id"]))


def _encode_portfolio(state: PortfolioCoordinatorState) -> dict[str, Any]:
    return {
        "revision": state.revision,
        "strategy_revisions": dict(state.strategy_revisions),
        "known_keys": [_account_key(key) for key in sorted(state.known_keys, key=str)],
        "contributions": {
            strategy_id: [
                {"key": _account_key(key), "quantity": str(quantity)}
                for key, quantity in sorted(values.items(), key=lambda item: str(item[0]))
            ]
            for strategy_id, values in sorted(state.contributions.items())
        },
    }


def _decode_portfolio(value: Any) -> PortfolioCoordinatorState:
    if not isinstance(value, dict):
        raise StateCorruptionError("portfolio格式无效")
    return PortfolioCoordinatorState(
        revision=value["revision"],
        contributions={
            strategy_id: {
                _decode_account_key(item["key"]): Decimal(item["quantity"])
                for item in items
            }
            for strategy_id, items in value["contributions"].items()
        },
        strategy_revisions=value["strategy_revisions"],
        known_keys=frozenset(_decode_account_key(item) for item in value["known_keys"]),
    )


def _encode_positions(state: PositionManagerState) -> dict[str, Any]:
    snapshot = state.snapshot
    return {
        "strategy_positions": [
            {"strategy_id": key[0], "target_key": key[1], "quantity": str(quantity)}
            for key, quantity in sorted(snapshot.strategy_positions.items())
        ],
        "account_positions": [
            {"key": _account_key(key), "quantity": str(quantity)}
            for key, quantity in sorted(snapshot.account_positions.items(), key=lambda item: str(item[0]))
        ],
        "working_quantities": [
            {"key": _account_key(key), "quantity": str(quantity)}
            for key, quantity in sorted(snapshot.working_quantities.items(), key=lambda item: str(item[0]))
        ],
        "account_reconciliations": {
            client_id: {"revision": item.revision, "ts_event": item.ts_event}
            for client_id, item in sorted(snapshot.account_reconciliations.items())
        },
        "recovery_required_clients": sorted(snapshot.recovery_required_clients),
        "account_revisions": [
            {"key": _account_key(key), "revision": revision}
            for key, revision in sorted(state.account_revisions.items(), key=lambda item: str(item[0]))
        ],
        "strategy_revisions": [
            {"strategy_id": key[0], "target_key": key[1], "revision": revision}
            for key, revision in sorted(state.strategy_revisions.items())
        ],
    }


def _decode_positions(value: Any) -> PositionManagerState:
    if not isinstance(value, dict):
        raise StateCorruptionError("positions格式无效")
    snapshot = PositionSnapshot(
        strategy_positions={
            (item["strategy_id"], item["target_key"]): Decimal(item["quantity"])
            for item in value["strategy_positions"]
        },
        account_positions={
            _decode_account_key(item["key"]): Decimal(item["quantity"])
            for item in value["account_positions"]
        },
        working_quantities={
            _decode_account_key(item["key"]): Decimal(item["quantity"])
            for item in value["working_quantities"]
        },
        account_reconciliations={
            client_id: AccountReconciliationState(item["revision"], item["ts_event"])
            for client_id, item in value["account_reconciliations"].items()
        },
        recovery_required_clients=frozenset(value.get("recovery_required_clients", ())),
    )
    return PositionManagerState(
        snapshot=snapshot,
        account_revisions={
            _decode_account_key(item["key"]): item["revision"]
            for item in value["account_revisions"]
        },
        strategy_revisions={
            (item["strategy_id"], item["target_key"]): item["revision"]
            for item in value["strategy_revisions"]
        },
    )


def _encode_orders(checkpoints: tuple[OrderStateCheckpoint, ...]) -> list[dict[str, Any]]:
    return [
        {
            "backend_id": item.state.backend_id,
            "client_order_id": item.state.client_order_id,
            "instrument_id": str(item.state.instrument_id),
            "side": item.state.side.value,
            "order_quantity": str(item.state.order_quantity),
            "filled_quantity": str(item.state.filled_quantity),
            "remaining_quantity": str(item.state.remaining_quantity),
            "status": item.state.status.value,
            "last_sequence": item.state.last_sequence,
            "last_ts_event": item.state.last_ts_event,
            "seen_keys": list(item.seen_keys),
        }
        for item in sorted(checkpoints, key=lambda value: value.state.client_order_id)
    ]


def _decode_orders(value: Any) -> tuple[OrderStateCheckpoint, ...]:
    if not isinstance(value, list):
        raise StateCorruptionError("orders格式无效")
    return tuple(
        OrderStateCheckpoint(
            state=OrderState(
                backend_id=item["backend_id"],
                client_order_id=item["client_order_id"],
                instrument_id=InstrumentId.from_str(item["instrument_id"]),
                side=OrderSide(item["side"]),
                order_quantity=Decimal(item["order_quantity"]),
                filled_quantity=Decimal(item["filled_quantity"]),
                remaining_quantity=Decimal(item["remaining_quantity"]),
                status=OrderLifecycleStatus(item["status"]),
                last_sequence=item["last_sequence"],
                last_ts_event=item["last_ts_event"],
            ),
            seen_keys=tuple(item["seen_keys"]),
        )
        for item in value
    )


def _encode_ctp_driver(checkpoint: CtpDriverCheckpoint) -> dict[str, Any]:
    return {
        "driver_id": checkpoint.driver_id,
        "account_id": checkpoint.account_id,
        "broker_id": checkpoint.broker_id,
        "investor_id": checkpoint.investor_id,
        "trading_day": checkpoint.trading_day,
        "next_ref": checkpoint.next_ref,
        "orders": [
            {
                "order_ref": item.order_ref,
                "client_order_id": item.client_order_id,
                "filled_quantity": str(item.filled_quantity),
                "sequence": item.sequence,
                "accepted": item.accepted,
                "seen_trades": [list(key) for key in item.seen_trades],
                "intent": {
                    "strategy_id": item.intent.strategy_id,
                    "backend_id": item.intent.backend_id,
                    "instrument_id": str(item.intent.instrument_id),
                    "side": item.intent.side.value,
                    "quantity": str(item.intent.quantity),
                    "order_type": item.intent.order_type.value,
                    "price": None if item.intent.price is None else str(item.intent.price),
                    "position_effect": item.intent.position_effect.value,
                    "reduce_only": item.intent.reduce_only,
                    "metadata": _json_value(item.intent.metadata),
                },
            }
            for item in checkpoint.orders
        ],
    }


def _decode_ctp_driver(value: Any) -> CtpDriverCheckpoint:
    if not isinstance(value, dict) or not isinstance(value.get("orders"), list):
        raise StateCorruptionError("CTP Driver检查点格式无效")
    records = []
    for item in value["orders"]:
        raw = item["intent"]
        records.append(CtpOrderAssociation(
            order_ref=item["order_ref"],
            client_order_id=item["client_order_id"],
            intent=OrderIntent(
                strategy_id=raw["strategy_id"], backend_id=raw["backend_id"],
                instrument_id=InstrumentId.from_str(raw["instrument_id"]),
                side=raw["side"], quantity=Decimal(raw["quantity"]),
                order_type=raw["order_type"],
                price=None if raw["price"] is None else Decimal(raw["price"]),
                position_effect=raw["position_effect"],
                reduce_only=raw["reduce_only"],
                metadata=_restore_json_value(raw["metadata"]),
            ),
            filled_quantity=Decimal(item["filled_quantity"]),
            sequence=item["sequence"], accepted=item["accepted"],
            seen_trades=tuple(tuple(key) for key in item["seen_trades"]),
        ))
    return CtpDriverCheckpoint(
        driver_id=value["driver_id"], account_id=value["account_id"],
        broker_id=value["broker_id"], investor_id=value["investor_id"],
        trading_day=value["trading_day"], next_ref=value["next_ref"],
        orders=tuple(records),
    )


def _encode_ctp_ledger(state: CtpLedgerState) -> dict[str, Any]:
    return {
        "trading_day": state.trading_day,
        "positions": [
            {
                "instrument_id": str(item.instrument_id),
                "long_today": str(item.long_today),
                "long_yesterday": str(item.long_yesterday),
                "short_today": str(item.short_today),
                "short_yesterday": str(item.short_yesterday),
                "long_today_basis": str(item.long_today_basis),
                "long_yesterday_basis": str(item.long_yesterday_basis),
                "short_today_basis": str(item.short_today_basis),
                "short_yesterday_basis": str(item.short_yesterday_basis),
            }
            for _, item in sorted(state.positions.items(), key=lambda value: str(value[0]))
        ],
    }


def _decode_ctp_ledger(value: Any) -> CtpLedgerState:
    if not isinstance(value, dict):
        raise StateCorruptionError("ctp_ledger格式无效")
    positions: dict[InstrumentId, CtpPositionSnapshot] = {}
    for item in value["positions"]:
        instrument_id = InstrumentId.from_str(item["instrument_id"])
        positions[instrument_id] = CtpPositionSnapshot(
            instrument_id=instrument_id,
            long_today=Decimal(item["long_today"]),
            long_yesterday=Decimal(item["long_yesterday"]),
            short_today=Decimal(item["short_today"]),
            short_yesterday=Decimal(item["short_yesterday"]),
            long_today_basis=Decimal(item["long_today_basis"]),
            long_yesterday_basis=Decimal(item["long_yesterday_basis"]),
            short_today_basis=Decimal(item["short_today_basis"]),
            short_yesterday_basis=Decimal(item["short_yesterday_basis"]),
        )
    return CtpLedgerState(value["trading_day"], positions)
