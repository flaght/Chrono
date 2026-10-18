"""Binance USDT单向期货的直接HTTP权威仓位/资金读取；不使用Cache。"""

import asyncio
from decimal import Decimal
import time

from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.trader.execution.events import AccountStateEvent, CurrencyBalance
from .binance_orders import _field


class BinanceNativeAccountBinding:
    def __init__(self):
        self.api = None

    def capture(self, native_client):
        api = getattr(native_client, "_futures_http_account", None)
        if self.api is not None or any(not callable(getattr(api, name, None)) for name in (
                "query_futures_account_info", "query_futures_position_risk", "query_futures_hedge_mode")):
            raise RuntimeError("原生Binance版本缺少全账户HTTP查询，或重复绑定")
        self.api = api

    async def positions(self):
        if self.api is None:
            raise RuntimeError("Binance HTTP账户尚未绑定")
        mode = await self.api.query_futures_hedge_mode()
        if _field(mode, "dualSidePosition") is not False:
            raise RuntimeError("联合首期只支持Binance单向账户，不允许对冲模式")
        return await self.api.query_futures_position_risk(None)

    async def account(self):
        if self.api is None:
            raise RuntimeError("Binance HTTP账户尚未绑定")
        return await self.api.query_futures_account_info()


class BinanceHttpAccountReader:
    def __init__(self, client_id, account_id, loop_provider, binding, *, timeout_seconds=15):
        if not client_id.strip() or not account_id.strip() or timeout_seconds <= 0:
            raise ValueError("账户查询身份与超时无效")
        self.client_id, self.account_id = client_id, account_id
        self.loop_provider, self.binding = loop_provider, binding
        self.timeout_seconds = timeout_seconds
        self.revision = 0

    def _read(self, operation):
        loop = self.loop_provider()
        if loop is None or not loop.is_running():
            raise RuntimeError("TradingNode事件循环未运行")
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is loop:
            raise RuntimeError("HTTP同步查询不能在TradingNode事件循环内调用")
        future = asyncio.run_coroutine_threadsafe(operation(), loop)
        try:
            return future.result(timeout=self.timeout_seconds)
        except BaseException:
            future.cancel()
            raise

    @staticmethod
    def parse_positions(rows):
        if not isinstance(rows, (list, tuple)):
            raise TypeError("HTTP仓位必须是全账户序列")
        positions, seen = {}, set()
        for row in rows:
            symbol = str(_field(row, "symbol")).upper()
            if not symbol.isalnum() or symbol in seen or _field(row, "positionSide") != "BOTH":
                raise ValueError("HTTP仓位合约重复、无效或非单向")
            seen.add(symbol)
            quantity = Decimal(str(_field(row, "positionAmt")))
            if not quantity.is_finite():
                raise ValueError("HTTP仓位数量必须有限")
            if quantity:
                positions[InstrumentId.from_str(f"{symbol}-PERP.BINANCE")] = quantity
        return positions

    def positions(self):
        return self.parse_positions(self._read(self.binding.positions))

    def account(self):
        raw = self._read(self.binding.account)
        if _field(raw, "canTrade") is not True:
            raise RuntimeError("Binance账户没有交易权限")
        balances = {}
        for asset in _field(raw, "assets"):
            currency = str(_field(asset, "asset"))
            if currency in balances:
                raise ValueError("HTTP资金币种重复")
            balances[currency] = CurrencyBalance(currency,
                total=_field(asset, "walletBalance"), equity=_field(asset, "marginBalance"),
                available=_field(asset, "availableBalance"), margin_used=_field(asset, "initialMargin"))
        state = AccountStateEvent(self.client_id, self.account_id, self.revision + 1,
            time.time_ns(), balances)
        self.revision = state.revision
        return state
