"""从Nautilus账户的交易所回报事件生成统一只读资金快照。"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Callable

from strategy.execution.events import AccountStateEvent, CurrencyBalance


def _number(value: Any) -> Decimal:
    if hasattr(value, "as_decimal"):
        value = value.as_decimal()
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("账户资金包含非有限数值")
    return result


class NautilusReportedAccountReader:
    """主动查询并等待一次*新的*交易所账户事件，拒绝缓存旧值。

    `query_account`由Driver在TradingNode事件循环中发起。读取的`account`
    必须是该节点的真实账户对象；不能用回测账户或自造事件替代。
    """

    def __init__(
        self,
        client_id: str,
        account_id: str,
        account_provider: Callable[[], Any],
        query_account: Callable[[Any], None],
        *,
        required_info_keys: tuple[str, ...],
        timeout_seconds: float = 10.0,
        poll_seconds: float = 0.05,
    ) -> None:
        if not client_id.strip() or not account_id.strip():
            raise ValueError("client_id和account_id不能为空")
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("账户查询超时和轮询间隔必须大于零")
        if not required_info_keys or any(not key.strip() for key in required_info_keys):
            raise ValueError("必须指定HTTP查询回报独有的info键以排除WebSocket缓存事件")
        self.client_id = client_id
        self.account_id = account_id
        self.account_provider = account_provider
        self.query_account = query_account
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.required_info_keys = required_info_keys
        self._revision = 0
        self._native_account_id: str | None = None

    def read(self) -> AccountStateEvent:
        account = self.account_provider()
        if account is None:
            raise RuntimeError("Nautilus账户尚未建立")
        native_id = str(account.id)
        if self._native_account_id is not None and native_id != self._native_account_id:
            raise RuntimeError("Nautilus账户ID发生变化，必须重新装配客户端")
        previous_ids = {str(event.id) for event in account.events if event.is_reported}
        self.query_account(account.id)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            current = self.account_provider()
            if current is None or str(current.id) != native_id:
                raise RuntimeError("账户查询期间原生账户已断开或改变")
            for event in reversed(current.events):
                info = event.info or {}
                if (event.is_reported and str(event.id) not in previous_ids
                        and all(key in info for key in self.required_info_keys)):
                    self._revision += 1
                    self._native_account_id = native_id
                    return self._convert(event)
            if time.monotonic() >= deadline:
                raise TimeoutError("等待新的交易所账户资金回报超时")
            time.sleep(self.poll_seconds)

    def _convert(self, event: Any) -> AccountStateEvent:
        info = event.info or {}
        balances: dict[str, CurrencyBalance] = {}
        for item in event.balances:
            code = str(item.currency.code)
            equity = None
            margin_used = None
            if code == "USDT":
                if info.get("total_margin_balance") is not None:
                    equity = _number(info["total_margin_balance"])
                if info.get("total_initial_margin") is not None:
                    margin_used = _number(info["total_initial_margin"])
            available = (
                _number(info["available_balance"])
                if code == "USDT" and info.get("available_balance") is not None
                else _number(item.free)
            )
            balances[code] = CurrencyBalance(
                code,
                total=_number(item.total),
                equity=equity,
                available=available,
                margin_used=margin_used,
            )
        return AccountStateEvent(
            client_id=self.client_id,
            account_id=self.account_id,
            revision=self._revision,
            ts_event=int(event.ts_event),
            balances=balances,
        )
