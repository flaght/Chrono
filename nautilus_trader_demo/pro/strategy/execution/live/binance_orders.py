"""Binance USDT期货活动订单的权威HTTP查询与标准快照转换。"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any, Awaitable, Callable, Sequence

from strategy.execution.events import ActiveOrder, ActiveOrderSnapshot


def _field(order: Any, name: str) -> Any:
    value = order.get(name) if isinstance(order, dict) else getattr(order, name, None)
    if value is None or value == "":
        raise ValueError(f"Binance活动订单缺少{name}")
    return getattr(value, "value", value)


class BinanceNativeOpenOrdersBinding:
    """把Factory创建的原生执行客户端绑定到HTTP查询边界。

    原生执行客户端没有公开的全市场open-orders方法；因此唯一的版本相关
    访问被限制在这里。Factory创建客户端后调用`capture`，版本不兼容时立即
    报错，绝不退化为Cache或会吞错的generate_order_status_reports。
    """

    def __init__(self) -> None:
        self._query: Callable[[None], Awaitable[Sequence[Any]]] | None = None
        self._query_algo: Callable[[None], Awaitable[Sequence[Any]]] | None = None

    def capture(self, native_client: Any) -> None:
        if self._query is not None:
            raise RuntimeError("Binance原生执行客户端已经绑定")
        account_api = getattr(native_client, "_futures_http_account", None)
        query = getattr(account_api, "query_open_orders", None)
        query_algo = getattr(account_api, "query_open_algo_orders", None)
        if not callable(query) or not callable(query_algo):
            raise RuntimeError("当前Binance客户端版本未提供完整的HTTP活动订单查询")
        self._query = query
        self._query_algo = query_algo

    async def query_open_orders(self) -> Sequence[Any]:
        if self._query is None or self._query_algo is None:
            raise RuntimeError("Binance HTTP活动订单查询尚未绑定")
        # 传None表示全市场，不能仅查BTC而遗漏账户其他合约的订单。
        regular = await self._query(None)
        algo = await self._query_algo(None)
        if algo is None or len(algo) != 0:
            # Algo订单尚无标准映射；不得把它们遗漏后宣称柜台无活动订单。
            raise RuntimeError("Binance存在未适配的Algo活动订单，保持禁单")
        return regular


class BinanceHttpActiveOrderReader:
    """在TradingNode事件循环上执行一次全市场HTTP open-orders查询。

    `query_open_orders`必须绑定原生Binance HTTP账户客户端的无symbol查询，
    不能传入Nautilus Cache或会把HTTP异常吞掉并返回空列表的报告生成器。
    返回空列表只有在这次HTTP调用成功完成后才代表柜台确认无活动订单。
    """

    def __init__(
        self,
        client_id: str,
        account_id: str,
        loop_provider: Callable[[], asyncio.AbstractEventLoop],
        query_open_orders: Callable[[], Awaitable[Sequence[Any]]],
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not client_id.strip() or not account_id.strip():
            raise ValueError("client_id和account_id不能为空")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds必须大于零")
        self.client_id = client_id
        self.account_id = account_id
        self.loop_provider = loop_provider
        self.query_open_orders = query_open_orders
        self.timeout_seconds = timeout_seconds
        self._revision = 0

    def read(self) -> ActiveOrderSnapshot:
        loop = self.loop_provider()
        if loop is None or not loop.is_running():
            raise RuntimeError("TradingNode事件循环未运行")
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            raise RuntimeError("活动订单同步查询不能在TradingNode事件循环线程调用")
        future = asyncio.run_coroutine_threadsafe(self.query_open_orders(), loop)
        try:
            raw_orders = future.result(timeout=self.timeout_seconds)
        except BaseException:
            future.cancel()
            raise
        if not isinstance(raw_orders, (list, tuple)):
            raise TypeError("Binance HTTP活动订单查询必须返回完整订单序列")
        orders: list[ActiveOrder] = []
        for raw in raw_orders:
            status = str(_field(raw, "status")).upper()
            if status not in {"NEW", "PARTIALLY_FILLED"}:
                raise ValueError(f"open-orders查询出现非活动状态: {status}")
            symbol = str(_field(raw, "symbol")).upper()
            side = str(_field(raw, "side")).upper()
            if not symbol.isalnum() or side not in {"BUY", "SELL"}:
                raise ValueError("Binance活动订单合约或方向无效")
            quantity = Decimal(str(_field(raw, "origQty")))
            filled = Decimal(str(_field(raw, "executedQty")))
            orders.append(ActiveOrder(
                client_order_id=str(_field(raw, "clientOrderId")),
                instrument_id=f"{symbol}-PERP.BINANCE",
                side=side,
                order_quantity=quantity,
                cumulative_filled=filled,
                remaining_quantity=quantity - filled,
            ))
        snapshot = ActiveOrderSnapshot(
            client_id=self.client_id,
            account_id=self.account_id,
            revision=self._revision + 1,
            ts_event=time.time_ns(),
            orders=tuple(orders),
        )
        self._revision = snapshot.revision
        return snapshot
