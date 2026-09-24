"""CTP TdApi的请求/回调聚合；默认只读，不负责开放交易授权。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping

from market.basic.base import InstrumentId
from trader.execution.ctp.native_driver import CtpTraderSession
from trader.execution.events import (
    AccountStateEvent,
    ActiveOrder,
    ActiveOrderSnapshot,
    CurrencyBalance,
)


@dataclass
class _Pending:
    kind: str
    reqid: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    done: bool = False


class CtpTdApiTransport:
    """将原生TdApi的分片OnRsp聚合为一次权威查询结果。

    `td_api_base`仅供假API测试；实际连接延迟导入项目自有的
    `bomber_ctp_td.TdApi`，不依赖vn.py交易Gateway。
    认证/登录/结算确认及所有查询失败一律抛异常，不返回旧缓存。
    """

    def __init__(
        self,
        *,
        client_id: str,
        account_id: str,
        front: str,
        broker_id: str,
        investor_id: str,
        password: str,
        app_id: str,
        auth_code: str,
        td_api_base: type | None = None,
        flow_path: str = "",
        production_mode: bool = True,
        timeout_seconds: float = 10.0,
    ) -> None:
        if any(not value.strip() for value in (
            client_id, account_id, front, broker_id, investor_id, password,
        )):
            raise ValueError("CTP连接标识、前置和账户凭据不能为空")
        if timeout_seconds <= 0:
            raise ValueError("CTP请求超时必须大于零")
        self.client_id = client_id
        self.account_id = account_id
        self.front = front
        self.broker_id = broker_id
        self.investor_id = investor_id
        self.password = password
        self.app_id = app_id
        self.auth_code = auth_code
        self.flow_path = flow_path
        self.production_mode = production_mode
        self.timeout_seconds = timeout_seconds
        self.td_api_base = td_api_base
        self._api: Any = None
        self._api_created = False
        self._api_initialized = False
        self._cv = threading.Condition()
        self._request_lock = threading.Lock()
        self._reqid = 0
        self._pending: _Pending | None = None
        self._front_connected = False
        self._disconnected = False
        self._ready = False
        self._session: CtpTraderSession | None = None
        self._early_pushes = 0
        self._on_order: Callable[[Mapping[str, Any]], None] | None = None
        self._on_trade: Callable[[Mapping[str, Any]], None] | None = None
        self._on_disconnect: Callable[[int], None] | None = None
        self._account_revision = 0
        self._orders_revision = 0
        self._last_orders: dict[str, dict[str, Any]] = {}
        self._insert_requests: dict[int, dict[str, Any]] = {}
        self._rejected_refs: set[str] = set()
        self._action_requests: dict[int, str] = {}

    def connect(self, on_order, on_trade, on_disconnect) -> CtpTraderSession:
        if self._api is not None:
            raise RuntimeError("CTP TdApi已创建，不能重复连接")
        self._on_order = on_order
        self._on_trade = on_trade
        self._on_disconnect = on_disconnect
        self._front_connected = False
        self._disconnected = False
        self._ready = False
        self._early_pushes = 0
        self._last_orders.clear()
        self._insert_requests.clear()
        self._rejected_refs.clear()
        self._action_requests.clear()
        base = self.td_api_base
        if base is None:
            try:
                from bomber_ctp_td import TdApi
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "未找到项目自有CTP交易扩展 bomber_ctp_td；"
                    "请先安装 market/native/ctp/binding"
                ) from exc
            base = TdApi
        owner = self

        class CallbackApi(base):
            def onFrontConnected(self):
                owner._front_up()

            def onFrontDisconnected(self, reason):
                owner._front_down(reason)

            def onRspError(self, error, reqid, last):
                # 柜台也可能只返回通用错误，不能让请求一直等待超时。
                owner._front_down(-int((error or {}).get("ErrorID", 1) or 1))

            def onRspAuthenticate(self, data, error, reqid, last):
                owner._response("auth", data, error, reqid, last)

            def onRspUserLogin(self, data, error, reqid, last):
                owner._response("login", data, error, reqid, last)

            def onRspSettlementInfoConfirm(self, data, error, reqid, last):
                owner._response("settlement", data, error, reqid, last)

            def onRspQryInvestorPosition(self, data, error, reqid, last):
                owner._response("positions", data, error, reqid, last)

            def onRspQryTradingAccount(self, data, error, reqid, last):
                owner._response("account", data, error, reqid, last)

            def onRspQryOrder(self, data, error, reqid, last):
                owner._response("orders", data, error, reqid, last)

            def onRtnOrder(self, data):
                owner._order_push(data)

            def onRtnTrade(self, data):
                owner._trade_push(data)

            def onRspOrderInsert(self, data, error, reqid, last):
                owner._insert_error(data, error, reqid)

            def onErrRtnOrderInsert(self, data, error):
                owner._insert_error(data, error, None)

            def onRspOrderAction(self, data, error, reqid, last):
                owner._action_error(data, error, reqid)

            def onErrRtnOrderAction(self, data, error):
                owner._action_error(data, error, None)

        api = CallbackApi()
        self._api = api
        try:
            flow_path = Path(self.flow_path or "/tmp/bomber-ctp-td").expanduser()
            flow_path.mkdir(parents=True, exist_ok=True)
            api.createFtdcTraderApi(f"{flow_path}/", self.production_mode)
            self._api_created = True
            api.subscribePrivateTopic(2)
            api.subscribePublicTopic(2)
            api.registerFront(self.front)
            api.init()
            self._api_initialized = True
            deadline = time.monotonic() + self.timeout_seconds
            with self._cv:
                while not self._front_connected and not self._disconnected:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("等待CTP交易前置连接超时")
                    self._cv.wait(remaining)
                if self._disconnected:
                    raise RuntimeError("CTP交易前置在认证前断开")
            if self.auth_code:
                self._request("auth", "reqAuthenticate", {
                    "BrokerID": self.broker_id, "UserID": self.investor_id,
                    "AuthCode": self.auth_code, "AppID": self.app_id,
                })
            rows = self._request("login", "reqUserLogin", {
                "BrokerID": self.broker_id, "UserID": self.investor_id,
                "Password": self.password,
            })
            if len(rows) != 1:
                raise RuntimeError("CTP登录未返回唯一会话")
            login = rows[0]
            if (str(login.get("BrokerID", "")) != self.broker_id
                    or str(login.get("UserID", "")) != self.investor_id):
                raise RuntimeError("CTP登录回报账户身份不匹配")
            session = CtpTraderSession(
                self.broker_id, self.investor_id,
                str(login.get("TradingDay", "")),
                int(login["MaxOrderRef"]),
            )
            confirmation = self._request("settlement", "reqSettlementInfoConfirm", {
                "BrokerID": self.broker_id, "InvestorID": self.investor_id,
            })
            if (len(confirmation) != 1
                    or str(confirmation[0].get("BrokerID", "")) != self.broker_id
                    or str(confirmation[0].get("InvestorID", "")) != self.investor_id):
                raise RuntimeError("CTP结算确认回报缺失或账户不匹配")
            with self._cv:
                if self._disconnected or self._early_pushes:
                    raise RuntimeError("CTP连接期间断线或出现未归属订单，保持禁单")
                self._session = session
            return session
        except Exception:
            self.close()
            raise

    def activate(self) -> None:
        """Driver建好会话映射和回报接收器后才允许柜台推送与权威查询。"""
        with self._cv:
            if self._api is None or self._session is None or self._disconnected or self._early_pushes:
                raise RuntimeError("CTP传输不可激活：会话断开或存在早到未归属推送")
            self._ready = True

    def close(self) -> None:
        with self._cv:
            self._ready = False
            self._disconnected = True
            self._session = None
            self._pending = None
            self._insert_requests.clear()
            self._rejected_refs.clear()
            self._action_requests.clear()
            self._cv.notify_all()
        api, self._api = self._api, None
        if api is not None:
            try:
                if self._api_initialized:
                    api.exit()
                elif self._api_created:
                    api.release()
            finally:
                self._api_initialized = False
                self._api_created = False

    def _require_ready(self) -> None:
        if not self._ready or self._disconnected or self._api is None:
            raise RuntimeError("CTP TraderApi尚未完成登录结算确认或已断线")

    def _request(self, kind: str, method: str, fields: dict[str, Any]) -> list[dict[str, Any]]:
        with self._request_lock:
            with self._cv:
                if self._disconnected or self._api is None:
                    raise RuntimeError("CTP查询前置已断开")
                self._reqid += 1
                pending = _Pending(kind, self._reqid)
                self._pending = pending
            code = getattr(self._api, method)(fields, pending.reqid)
            if code != 0:
                with self._cv:
                    self._pending = None
                raise RuntimeError(f"CTP {method}请求被拒绝: {code}")
            deadline = time.monotonic() + self.timeout_seconds
            with self._cv:
                while not pending.done and not self._disconnected:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._pending = None
                        raise TimeoutError(f"等待CTP {kind}最终回报超时")
                    self._cv.wait(remaining)
                self._pending = None
                if self._disconnected or pending.error:
                    raise RuntimeError(pending.error or f"CTP {kind}查询期间前置断线")
                return pending.rows

    def _front_up(self) -> None:
        with self._cv:
            self._front_connected = True
            self._cv.notify_all()

    def _front_down(self, reason: int) -> None:
        with self._cv:
            was_ready = self._ready
            self._disconnected = True
            self._ready = False
            self._cv.notify_all()
        if was_ready and self._on_disconnect is not None:
            self._on_disconnect(reason)

    def _response(self, kind: str, data: Any, error: Any, reqid: int, last: bool) -> None:
        with self._cv:
            pending = self._pending
            if pending is None or pending.kind != kind or pending.reqid != reqid:
                # 超时请求的迟到回报不能冒充下一次查询。
                return
            error_info = error or {}
            error_id = int(error_info.get("ErrorID", 0))
            if error_id:
                message = str(error_info.get("ErrorMsg") or "").strip()
                pending.error = (
                    f"CTP {kind}回报错误: {error_id}"
                    + (f" ({message})" if message else "")
                )
            if data and not pending.error:
                pending.rows.append(dict(data))
            if last or pending.error:
                pending.done = True
                self._cv.notify_all()

    def _order_push(self, data: Any) -> None:
        row = dict(data)
        with self._cv:
            if not self._ready:
                self._early_pushes += 1
                return
            self._last_orders[str(row.get("OrderRef", ""))] = row
        if self._on_order is not None:
            self._on_order(row)

    def _trade_push(self, data: Any) -> None:
        row = dict(data)
        with self._cv:
            if not self._ready:
                self._early_pushes += 1
                return
        if self._on_trade is not None:
            self._on_trade(row)

    def _insert_error(self, data: Any, error: Any, reqid: int | None) -> None:
        error_id = int((error or {}).get("ErrorID", 0))
        if not error_id:
            return
        row = dict(data or {})
        with self._cv:
            if str(row.get("OrderRef", "")) in self._rejected_refs:
                return
            request = self._insert_requests.get(reqid) if reqid is not None else None
            if request is None:
                ref = str(row.get("OrderRef", ""))
                request = next((item for item in self._insert_requests.values()
                                if str(item.get("OrderRef", "")) == ref), None)
        if request is None:
            self._front_down(-error_id)
            return
        # 柜台错误回报可能只携带OrderRef；其余身份必须来自本次发送记录。
        identity = {key: request[key] for key in (
            "BrokerID", "InvestorID", "InstrumentID", "ExchangeID", "OrderRef",
        )}
        if any(key in row and str(row[key]) != str(value)
               for key, value in identity.items()):
            self._front_down(-error_id)
            return
        with self._cv:
            self._rejected_refs.add(str(identity["OrderRef"]))
            for key, item in tuple(self._insert_requests.items()):
                if str(item.get("OrderRef", "")) == str(identity["OrderRef"]):
                    self._insert_requests.pop(key, None)
        self._order_push({
            **identity, "OrderSubmitStatus": "4",
            "StatusMsg": f"CTP报单拒绝 {error_id}: {(error or {}).get('ErrorMsg', '')}",
        })

    def _action_error(self, data: Any, error: Any, reqid: int | None) -> None:
        error_id = int((error or {}).get("ErrorID", 0))
        if not error_id:
            return
        row = dict(data or {})
        with self._cv:
            ref = self._action_requests.get(reqid) if reqid is not None else None
            if ref is None:
                ref = str(row.get("OrderRef", ""))
            known = ref in self._last_orders
        # 撤单拒绝不等于已撤单。订单仍可能在柜台活动，必须闭闸重新查询。
        self._front_down(-error_id if known else -1)

    def query_positions(self) -> Mapping[InstrumentId | str, Decimal]:
        self._require_ready()
        rows = self._request("positions", "reqQryInvestorPosition", {
            "BrokerID": self.broker_id, "InvestorID": self.investor_id,
        })
        positions: dict[str, Decimal] = {}
        for row in rows:
            if (str(row.get("BrokerID", "")) != self.broker_id
                    or str(row.get("InvestorID", "")) != self.investor_id):
                raise RuntimeError("CTP仓位回报账户不匹配")
            symbol, venue = str(row.get("InstrumentID", "")), str(row.get("ExchangeID", ""))
            direction = str(row.get("PosiDirection", ""))
            if not symbol or not venue or direction not in {"2", "3"}:
                raise RuntimeError("CTP仓位回报缺少合约、交易所或多空方向")
            quantity = Decimal(str(row.get("Position", "")))
            if not quantity.is_finite() or quantity < 0:
                raise RuntimeError("CTP仓位数量无效")
            instrument = f"{symbol}.{venue}"
            signed = quantity if direction == "2" else -quantity
            positions[instrument] = positions.get(instrument, Decimal(0)) + signed
        return positions

    def query_gross_positions(self) -> Mapping[str, tuple[Decimal, Decimal]]:
        """本次柜台查询的逐合约多空总仓；用于初次启用前确认真正空仓。"""
        self._require_ready()
        rows = self._request("positions", "reqQryInvestorPosition", {
            "BrokerID": self.broker_id, "InvestorID": self.investor_id,
        })
        gross: dict[str, tuple[Decimal, Decimal]] = {}
        for row in rows:
            if (str(row.get("BrokerID", "")) != self.broker_id
                    or str(row.get("InvestorID", "")) != self.investor_id):
                raise RuntimeError("CTP仓位回报账户不匹配")
            symbol = str(row.get("InstrumentID", ""))
            venue = str(row.get("ExchangeID", ""))
            direction = str(row.get("PosiDirection", ""))
            if not symbol or not venue or direction not in {"2", "3"}:
                raise RuntimeError("CTP仓位回报身份或方向缺失")
            quantity = Decimal(str(row.get("Position", "")))
            if not quantity.is_finite() or quantity < 0:
                raise RuntimeError("CTP仓位数量无效")
            key = f"{symbol}.{venue}"
            long_qty, short_qty = gross.get(key, (Decimal(0), Decimal(0)))
            gross[key] = (long_qty + quantity, short_qty) if direction == "2" else (
                long_qty, short_qty + quantity)
        return gross

    def query_account(self) -> AccountStateEvent:
        self._require_ready()
        rows = self._request("account", "reqQryTradingAccount", {
            "BrokerID": self.broker_id, "InvestorID": self.investor_id,
        })
        if not rows:
            raise RuntimeError("CTP资金查询没有返回账户记录")
        balances: dict[str, CurrencyBalance] = {}
        for row in rows:
            if (str(row.get("BrokerID", "")) != self.broker_id
                    or str(row.get("AccountID", "")) != self.investor_id):
                raise RuntimeError("CTP资金回报账户不匹配")
            currency = str(row.get("CurrencyID", ""))
            if not currency or any(row.get(name) is None for name in (
                "Balance", "Available", "CurrMargin",
            )):
                raise RuntimeError("CTP资金回报缺少币种、余额、可用或保证金字段")
            if currency in balances:
                raise RuntimeError("CTP资金查询返回重复币种")
            balances[currency] = CurrencyBalance(
                currency,
                total=row.get("Balance"),
                equity=row.get("Balance"),
                available=row.get("Available"),
                margin_used=row.get("CurrMargin"),
            )
        state = AccountStateEvent(
            self.client_id, self.account_id, self._account_revision + 1,
            time.time_ns(), balances,
        )
        self._account_revision = state.revision
        return state

    def query_active_orders(self) -> ActiveOrderSnapshot:
        self._require_ready()
        rows = self._request("orders", "reqQryOrder", {
            "BrokerID": self.broker_id, "InvestorID": self.investor_id,
        })
        session = self._session
        if session is None:
            raise RuntimeError("CTP活动订单查询缺少登录会话")
        orders: list[ActiveOrder] = []
        for row in rows:
            if (str(row.get("BrokerID", "")) != self.broker_id
                    or str(row.get("InvestorID", "")) != self.investor_id):
                raise RuntimeError("CTP订单回报账户不匹配")
            status = str(row.get("OrderStatus", ""))
            if status in {"0", "5"} or str(row.get("OrderSubmitStatus", "")) == "4":
                continue
            if status not in {"1", "2", "3", "4", "a", "b", "c"}:
                raise RuntimeError(f"CTP未知活动订单状态: {status}")
            order_ref = str(row.get("OrderRef", ""))
            symbol, venue = str(row.get("InstrumentID", "")), str(row.get("ExchangeID", ""))
            side = {"0": "BUY", "1": "SELL"}.get(str(row.get("Direction", "")))
            if not order_ref or not symbol or not venue or side is None:
                raise RuntimeError("CTP活动订单缺少身份字段")
            original = Decimal(str(row.get("VolumeTotalOriginal", "")))
            filled = Decimal(str(row.get("VolumeTraded", "")))
            remaining = Decimal(str(row.get("VolumeTotal", "")))
            orders.append(ActiveOrder(
                client_order_id=(
                    f"CTP-{self.broker_id}-{self.investor_id}-"
                    f"{session.trading_day}-{order_ref}"
                ),
                instrument_id=f"{symbol}.{venue}", side=side,
                order_quantity=original, cumulative_filled=filled,
                remaining_quantity=remaining,
            ))
        snapshot = ActiveOrderSnapshot(
            self.client_id, self.account_id, self._orders_revision + 1,
            time.time_ns(), tuple(orders),
        )
        self._orders_revision = snapshot.revision
        return snapshot

    def send_order(self, fields: Mapping[str, Any]) -> None:
        self._require_ready()
        request = dict(fields)
        reqid = self._next_reqid()
        with self._cv:
            self._insert_requests[reqid] = request
        code = self._api.reqOrderInsert(request, reqid)
        if code != 0:
            with self._cv:
                self._insert_requests.pop(reqid, None)
            raise RuntimeError(f"CTP reqOrderInsert请求被拒绝: {code}")

    def cancel_order(self, order_ref: str) -> None:
        self._require_ready()
        row = self._last_orders.get(order_ref)
        if row is None:
            raise RuntimeError("CTP撤单缺少柜台OrderRef与FrontID/SessionID映射")
        request = {
            "BrokerID": self.broker_id, "InvestorID": self.investor_id,
            "UserID": self.investor_id, "OrderRef": order_ref,
            "FrontID": row["FrontID"], "SessionID": row["SessionID"],
            "InstrumentID": row["InstrumentID"], "ExchangeID": row["ExchangeID"],
            "ActionFlag": "0",
        }
        reqid = self._next_reqid()
        with self._cv:
            self._action_requests[reqid] = order_ref
        code = self._api.reqOrderAction(request, reqid)
        if code != 0:
            with self._cv:
                self._action_requests.pop(reqid, None)
            raise RuntimeError(f"CTP reqOrderAction请求被拒绝: {code}")

    def _next_reqid(self) -> int:
        with self._cv:
            self._reqid += 1
            return self._reqid
