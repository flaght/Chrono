"""原生CTP TraderApi的执行Driver边界；底层传输由柜台适配器注入。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Mapping, Protocol

from market.basic.base import InstrumentId
from trader.execution.contracts import (
    AmbiguousOrderSubmission,
    ExecutionReport,
    ExecutionReportType,
    OrderIntent,
)
from trader.execution.ctp.native_order import make_ctp_order_insert
from trader.execution.events import AccountStateEvent, ActiveOrderSnapshot


@dataclass(frozen=True)
class CtpTraderSession:
    """已完成前置连接、认证、登录和结算确认后的会话。"""

    broker_id: str
    investor_id: str
    trading_day: str
    max_order_ref: int

    def __post_init__(self) -> None:
        if not self.broker_id or not self.investor_id or not self.trading_day:
            raise ValueError("CTP会话缺少经纪商、投资者或交易日")
        if self.max_order_ref < 0:
            raise ValueError("CTP最大OrderRef不能为负")


@dataclass(frozen=True)
class CtpOrderAssociation:
    """一笔未完成委托的本地归属与去重状态，必须与柜台活动订单核对。"""

    order_ref: str
    client_order_id: str
    intent: OrderIntent
    filled_quantity: Decimal
    sequence: int
    accepted: bool
    seen_trades: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class CtpDriverCheckpoint:
    """与订单状态机一起原子持久化；本身不能授权下单。"""

    driver_id: str
    account_id: str
    broker_id: str
    investor_id: str
    trading_day: str
    next_ref: int
    orders: tuple[CtpOrderAssociation, ...]


class CtpTraderTransport(Protocol):
    """实际TdApi绑定必须实现的同步权威查询与异步回调契约。

    每项查询必须等待本次ReqQry的最后一个OnRsp，检查全部ErrorID后返回；
    不得从本地缓存拼快照。`connect`只有完成结算确认后才能返回会话；此前
    到达的柜台推送必须由Transport缓冲，不得提前投递给尚未建好映射的Driver。
    """

    def connect(
        self,
        on_order: Callable[[Mapping[str, Any]], None],
        on_trade: Callable[[Mapping[str, Any]], None],
        on_disconnect: Callable[[int], None],
    ) -> CtpTraderSession: ...

    def activate(self) -> None: ...

    def close(self) -> None: ...

    def query_positions(self) -> Mapping[InstrumentId | str, Decimal | int | str]: ...

    def query_account(self) -> AccountStateEvent: ...

    def query_active_orders(self) -> ActiveOrderSnapshot: ...

    def send_order(self, fields: Mapping[str, Any]) -> None: ...

    def cancel_order(self, order_ref: str) -> None: ...


class CtpNativeTraderDriver:
    """CTP回调转标准执行回报；未显式解锁时只能连接和查询。"""

    def __init__(
        self,
        driver_id: str,
        account_id: str,
        transport: CtpTraderTransport,
        *,
        enable_test_orders: bool = False,
        enable_simnow_orders: bool = False,
        disconnect_handler: Callable[[str], None] | None = None,
    ) -> None:
        if not driver_id.strip() or not account_id.strip():
            raise ValueError("CTP Driver与账户ID不能为空")
        if (enable_test_orders or enable_simnow_orders) and disconnect_handler is None:
            raise ValueError("启用CTP报单必须配置断线闭闸回调")
        if enable_test_orders and enable_simnow_orders:
            raise ValueError("假柜台与SimNow报单不能同时开启")
        if enable_test_orders and not getattr(transport, "is_test_transport", False):
            raise ValueError("测试报单开关只允许假柜台传输；真实TdApi仍保持禁单")
        if enable_simnow_orders:
            if (
                getattr(transport, "is_test_transport", False)
                or getattr(transport, "broker_id", None) != "9999"
                or getattr(transport, "investor_id", None) != account_id
                or not str(getattr(transport, "front", "")).startswith("tcp://")
            ):
                raise ValueError("SimNow报单要求BrokerID=9999、匹配的账户及已配置的CTP交易前置")
        self.driver_id = driver_id
        self.account_id = account_id
        self.transport = transport
        self._enable_test_orders = enable_test_orders or enable_simnow_orders
        self._enable_simnow_orders = enable_simnow_orders
        self._disconnect_handler = disconnect_handler
        self._sink: Callable[[ExecutionReport], None] | None = None
        self._session: CtpTraderSession | None = None
        self._connected = False
        self._next_ref = 0
        self._orders: dict[str, tuple[str, OrderIntent]] = {}
        self._sequences: dict[str, int] = {}
        self._filled: dict[str, Decimal] = {}
        self._seen_trades: set[tuple[str, str, str]] = set()
        self._trade_refs: dict[tuple[str, str, str], str] = {}
        self._accepted: set[str] = set()
        self._completed_refs: set[str] = set()
        self._recovery_checkpoint: CtpDriverCheckpoint | None = None
        self._before_send: Callable[[str, OrderIntent], None] | None = None
        self._after_transition: Callable[[], None] | None = None
        self._on_unsent: Callable[[str], None] | None = None

    @property
    def is_simnow_session(self) -> bool:
        session = self._session
        return bool(
            self._enable_simnow_orders and self._connected
            and session is not None and session.broker_id == "9999"
            and session.investor_id == self.account_id
        )

    def bind_durability(
        self,
        before_send: Callable[[str, OrderIntent], None],
        after_transition: Callable[[], None],
        on_unsent: Callable[[str], None],
    ) -> None:
        """装配层提供同代状态保存；只能在Driver启动前绑定。"""
        if self._connected or self._before_send is not None:
            raise RuntimeError("CTP持久化钩子只能在启动前绑定一次")
        self._before_send = before_send
        self._after_transition = after_transition
        self._on_unsent = on_unsent

    def start(self, report_sink: Callable[[ExecutionReport], None]) -> None:
        if self._connected:
            return
        session = self.transport.connect(self.on_order, self.on_trade, self.on_disconnect)
        if not isinstance(session, CtpTraderSession):
            self.transport.close()
            raise TypeError("CTP传输必须返回已认证且结算确认的会话")
        self._session = session
        self._next_ref = max(self._next_ref, session.max_order_ref)
        self._sink = report_sink
        self._connected = True
        try:
            # 此前Transport只缓冲/拒绝推送；会话与回报接收器就绪后才放行。
            self.transport.activate()
        except Exception:
            self._connected = False
            self._session = None
            self._sink = None
            self.transport.close()
            raise

    def stop(self) -> None:
        try:
            self.transport.close()
        finally:
            self._connected = False
            self._session = None
            self._sink = None

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("CTP交易前置未连接或尚未完成登录结算确认")

    def reconcile(self) -> Mapping[InstrumentId | str, Decimal | int | str]:
        self._require_connected()
        positions = self.transport.query_positions()
        if positions is None:
            raise RuntimeError("CTP权威仓位查询未完成")
        return positions

    def reconcile_account_state(self) -> AccountStateEvent:
        self._require_connected()
        state = self.transport.query_account()
        if (not isinstance(state, AccountStateEvent)
                or state.account_id != self.account_id
                or state.client_id != self.driver_id):
            raise RuntimeError("CTP权威资金查询结果无效")
        return state

    def reconcile_active_orders(self) -> ActiveOrderSnapshot:
        self._require_connected()
        snapshot = self.transport.query_active_orders()
        if (not isinstance(snapshot, ActiveOrderSnapshot)
                or snapshot.client_id != self.driver_id
                or snapshot.account_id != self.account_id):
            raise RuntimeError("CTP权威活动订单查询结果无效")
        if self._recovery_checkpoint is not None:
            try:
                self._restore_associations_from_snapshot(self._recovery_checkpoint, snapshot)
            except Exception:
                self.on_disconnect(-1)
                raise
        return snapshot

    def checkpoint(self) -> CtpDriverCheckpoint:
        """导出当前活动委托关联；调用方须与订单状态机同代保存。"""
        session = self._session
        if session is None or not self._connected or self._recovery_checkpoint is not None:
            raise RuntimeError("CTP会话未完成关联恢复，不能生成检查点")
        records = tuple(
            CtpOrderAssociation(
                ref, client_id, order, self._filled[ref],
                self._sequences.get(ref, 0), ref in self._accepted,
                tuple(sorted(key for key, trade_ref in self._trade_refs.items()
                             if trade_ref == ref)),
            )
            for ref, (client_id, order) in sorted(self._orders.items())
        )
        return CtpDriverCheckpoint(
            self.driver_id, self.account_id, session.broker_id,
            session.investor_id, session.trading_day, self._next_ref, records,
        )

    def stage_recovery(self, checkpoint: CtpDriverCheckpoint) -> None:
        """启动前暂存本地关联；之后必须查询柜台全量活动订单。"""
        if self._connected or self._session is not None or self._orders:
            raise RuntimeError("CTP Driver运行中不能装载恢复检查点")
        if (not isinstance(checkpoint, CtpDriverCheckpoint)
                or checkpoint.driver_id != self.driver_id
                or checkpoint.account_id != self.account_id):
            raise ValueError("CTP检查点客户端或账户不匹配")
        self._recovery_checkpoint = checkpoint

    def discard_staged_recovery(self) -> None:
        """仅供原子恢复失败时回滚尚未激活的检查点。"""
        if self._connected or self._session is not None:
            raise RuntimeError("CTP Driver运行中不能撤销恢复检查点")
        self._recovery_checkpoint = None

    def _restore_associations_from_snapshot(
        self, checkpoint: CtpDriverCheckpoint, snapshot: ActiveOrderSnapshot,
    ) -> None:
        session = self._session
        if session is None or (
            checkpoint.broker_id != session.broker_id
            or checkpoint.investor_id != session.investor_id
            or checkpoint.trading_day != session.trading_day
        ):
            raise RuntimeError("CTP检查点账户或交易日与当前登录会话不一致")
        refs = [record.order_ref for record in checkpoint.orders]
        if len(refs) != len(set(refs)) or checkpoint.next_ref < 0:
            raise RuntimeError("CTP检查点OrderRef重复或序号无效")
        reported = {order.client_order_id: order for order in snapshot.orders}
        local = {record.client_order_id: record for record in checkpoint.orders}
        if len(local) != len(refs) or reported.keys() != local.keys():
            raise RuntimeError("CTP柜台与本地活动订单ID不一致，需要人工对账")
        restored_trades: dict[tuple[str, str, str], str] = {}
        for record in checkpoint.orders:
            expected_id = (
                f"CTP-{session.broker_id}-{session.investor_id}-"
                f"{session.trading_day}-{record.order_ref}"
            )
            order = reported[record.client_order_id]
            intent = record.intent
            if (
                not record.order_ref.isdigit()
                or int(record.order_ref) > checkpoint.next_ref
                or record.client_order_id != expected_id
                or intent.backend_id != self.driver_id
                or not intent.strategy_id
                or str(intent.instrument_id) != order.instrument_id
                or intent.side.value != order.side.value
                or intent.quantity != order.order_quantity
                or record.filled_quantity != order.cumulative_filled
                or intent.quantity - record.filled_quantity != order.remaining_quantity
                or record.sequence < 0
            ):
                raise RuntimeError(f"CTP活动订单{record.client_order_id}归属或数量不一致")
            for key in record.seen_trades:
                if len(key) != 3 or key[0] != session.trading_day or key in restored_trades:
                    raise RuntimeError("CTP检查点成交去重键无效或重复")
                restored_trades[key] = record.order_ref
        self._orders = {
            record.order_ref: (record.client_order_id, record.intent)
            for record in checkpoint.orders
        }
        self._filled = {
            record.order_ref: record.filled_quantity for record in checkpoint.orders
        }
        self._sequences = {
            record.order_ref: record.sequence for record in checkpoint.orders
        }
        self._accepted = {
            record.order_ref for record in checkpoint.orders if record.accepted
        }
        self._trade_refs = restored_trades
        self._seen_trades = set(restored_trades)
        self._next_ref = max(self._next_ref, checkpoint.next_ref)
        self._recovery_checkpoint = None

    def submit_order(self, order: OrderIntent) -> None:
        self._require_connected()
        if self._recovery_checkpoint is not None:
            raise RuntimeError("CTP活动订单关联尚未经过柜台权威快照恢复")
        if not self._enable_test_orders:
            raise RuntimeError("原生CTP Driver默认禁单；本阶段只读")
        if order.backend_id != self.driver_id:
            raise ValueError("CTP订单Backend不匹配")
        session = self._session
        if session is None:
            raise RuntimeError("CTP会话不存在")
        next_ref = self._next_ref + 1
        order_ref = str(next_ref)
        fields = make_ctp_order_insert(
            order, broker_id=session.broker_id,
            investor_id=session.investor_id, order_ref=order_ref,
        )
        client_id = (
            f"CTP-{session.broker_id}-{session.investor_id}-"
            f"{session.trading_day}-{order_ref}"
        )
        # 拒绝或同步回调可能在send_order返回前发生，所以先建关联。
        self._orders[order_ref] = (client_id, order)
        self._filled[order_ref] = Decimal(0)
        self._next_ref = next_ref
        try:
            if self._before_send is not None:
                self._before_send(client_id, order)
            self.transport.send_order(fields)
        except Exception as error:
            if self._sequences.get(order_ref, 0) > 0:
                self.on_disconnect(-1)
                raise AmbiguousOrderSubmission(
                    "CTP发送调用失败，但柜台回报已到达；必须重新对账",
                ) from error
            if order_ref in self._orders:
                self._orders.pop(order_ref, None)
                self._filled.pop(order_ref, None)
                if self._on_unsent is not None:
                    self._on_unsent(client_id)
            else:
                # 同步柜台回报已经改变订单状态，不能当作“未发送”回滚。
                self.on_disconnect(-1)
            raise

    def cancel_strategy(self, strategy_id: str) -> None:
        self._require_connected()
        if self._recovery_checkpoint is not None:
            raise RuntimeError("CTP活动订单关联尚未经过柜台权威快照恢复")
        if not self._enable_test_orders:
            raise RuntimeError("原生CTP Driver默认禁单；本阶段只读")
        for order_ref, (_, order) in tuple(self._orders.items()):
            if order.strategy_id == strategy_id:
                self.transport.cancel_order(order_ref)

    def on_disconnect(self, reason: int) -> None:
        self._connected = False
        if self._disconnect_handler is not None:
            self._disconnect_handler(f"ctp_front_disconnected:{reason}")

    def on_order(self, raw: Mapping[str, Any]) -> None:
        order_ref = str(raw.get("OrderRef", ""))
        if order_ref in self._completed_refs:
            return
        if order_ref not in self._orders:
            # 外部未知订单只能通过权威查询和人工对账处置，不能虚构策略归属。
            self.on_disconnect(-1)
            raise RuntimeError(f"CTP回报出现未知OrderRef: {order_ref}")
        self._check_identity(order_ref, raw)
        status = str(raw.get("OrderStatus", ""))
        if str(raw.get("OrderSubmitStatus", "")) == "4":
            self._emit(order_ref, ExecutionReportType.REJECTED, raw,
                       reason=str(raw.get("StatusMsg", "CTP报单拒绝")))
            return
        if status in {"3", "1", "2", "0"} and order_ref not in self._accepted:
            self._accepted.add(order_ref)
            self._emit(order_ref, ExecutionReportType.ACCEPTED, raw)
        if status == "5":
            self._emit(order_ref, ExecutionReportType.CANCELED, raw)

    def on_trade(self, raw: Mapping[str, Any]) -> None:
        order_ref = str(raw.get("OrderRef", ""))
        trade_id = str(raw.get("TradeID", ""))
        exchange = str(raw.get("ExchangeID", ""))
        if not trade_id or not exchange:
            self.on_disconnect(-1)
            raise RuntimeError("CTP成交缺少TradeID或ExchangeID")
        trading_day = self._session.trading_day if self._session is not None else ""
        key = (trading_day, exchange, trade_id)
        if key in self._seen_trades:
            return
        if order_ref not in self._orders:
            self.on_disconnect(-1)
            raise RuntimeError(f"CTP成交出现未知OrderRef: {order_ref}")
        self._check_identity(order_ref, raw)
        _, order = self._orders[order_ref]
        quantity = Decimal(str(raw.get("Volume", "0")))
        price = Decimal(str(raw.get("Price", "0")))
        cumulative = self._filled[order_ref] + quantity
        if quantity <= 0 or price <= 0 or cumulative > order.quantity:
            self.on_disconnect(-1)
            raise RuntimeError("CTP成交数量或价格与本地订单矛盾")
        if order_ref not in self._accepted:
            self._accepted.add(order_ref)
            self._emit(order_ref, ExecutionReportType.ACCEPTED, raw)
        self._seen_trades.add(key)
        self._trade_refs[key] = order_ref
        self._filled[order_ref] = cumulative
        self._emit(
            order_ref,
            ExecutionReportType.FILLED if cumulative == order.quantity
            else ExecutionReportType.PARTIALLY_FILLED,
            raw, quantity=quantity, price=price,
            trade_id=f"{exchange}:{trade_id}",
        )

    def _check_identity(self, order_ref: str, raw: Mapping[str, Any]) -> None:
        session = self._session
        if session is None:
            self.on_disconnect(-1)
            raise RuntimeError("CTP回报到达时会话不存在")
        _, order = self._orders[order_ref]
        symbol, _, venue = str(order.instrument_id).rpartition(".")
        required = {
            "BrokerID": session.broker_id,
            "InvestorID": session.investor_id,
            "InstrumentID": symbol,
            "ExchangeID": venue,
        }
        if any(str(raw.get(name, "")) != expected for name, expected in required.items()):
            self.on_disconnect(-1)
            raise RuntimeError("CTP回报账户或合约身份与本地订单不匹配")

    def _emit(
        self,
        order_ref: str,
        report_type: ExecutionReportType,
        raw: Mapping[str, Any],
        *,
        quantity: Decimal = Decimal(0),
        price: Decimal | None = None,
        trade_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        if self._sink is None:
            raise RuntimeError("CTP回报接收器未启动")
        client_id, order = self._orders[order_ref]
        sequence = self._sequences.get(order_ref, 0) + 1
        self._sequences[order_ref] = sequence
        metadata = {
            "strategy_id": order.strategy_id,
            "position_effect": order.position_effect.value,
            "order_ref": order_ref,
            "front_id": str(raw.get("FrontID", "")),
            "session_id": str(raw.get("SessionID", "")),
            "venue_order_id": str(raw.get("OrderSysID", "")),
        }
        if trade_id is not None:
            metadata["trade_id"] = trade_id
        report = ExecutionReport(
            backend_id=self.driver_id,
            client_order_id=client_id,
            instrument_id=order.instrument_id,
            report_type=report_type,
            ts_event=time.time_ns(),
            filled_quantity=quantity,
            fill_price=price,
            order_side=order.side,
            order_quantity=order.quantity,
            position_effect=order.position_effect,
            reason=reason,
            report_id=f"{client_id}:{sequence}",
            sequence=sequence,
            metadata=metadata,
        )
        try:
            if report_type in {
                ExecutionReportType.REJECTED,
                ExecutionReportType.CANCELED,
                ExecutionReportType.FILLED,
            }:
                self._orders.pop(order_ref, None)
                self._completed_refs.add(order_ref)
            self._sink(report)
            if self._after_transition is not None:
                self._after_transition()
        except Exception:
            self.on_disconnect(-1)
            raise
