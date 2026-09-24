"""P4-BN1：原生Binance HTTP活动订单查询的无网络验证。"""

from __future__ import annotations

import asyncio
import threading
from decimal import Decimal

from trader.execution.live.binance_orders import (
    BinanceHttpActiveOrderReader,
    BinanceNativeOpenOrdersBinding,
)


def main() -> None:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        binding = BinanceNativeOpenOrdersBinding()
        try:
            binding.capture(object())
        except RuntimeError as error:
            assert "HTTP活动订单" in str(error)
        else:
            raise AssertionError("不能以Cache或无HTTP能力的客户端替代柜台查询")

        class FakeHttpAccount:
            def __init__(self):
                self.queried_symbols = []
                self.algo_orders = []

            async def query_open_orders(self, symbol=None):
                self.queried_symbols.append(symbol)
                return []

            async def query_open_algo_orders(self, symbol=None):
                self.queried_symbols.append(("algo", symbol))
                return self.algo_orders

        account_api = FakeHttpAccount()
        native_client = type("NativeClient", (), {"_futures_http_account": account_api})()
        binding.capture(native_client)
        bound_reader = BinanceHttpActiveOrderReader(
            "binance-demo", "demo-account", lambda: loop, binding.query_open_orders,
        )
        assert bound_reader.read().orders == ()
        assert account_api.queried_symbols == [None, ("algo", None)]
        account_api.algo_orders = [object()]
        try:
            bound_reader.read()
        except RuntimeError as error:
            assert "Algo活动订单" in str(error)
        else:
            raise AssertionError("不能遗漏Binance Algo活动订单")
        print("P4-BN2a通过：普通/Algo订单均全市场查询，未知Algo订单保持禁单")

        async def query():
            return [{
                "symbol": "BTCUSDT", "clientOrderId": "O-1", "side": "BUY",
                "status": "PARTIALLY_FILLED", "origQty": "2", "executedQty": "0.5",
            }]

        reader = BinanceHttpActiveOrderReader(
            "binance-demo", "demo-account", lambda: loop, query,
        )
        snapshot = reader.read()
        assert snapshot.revision == 1 and len(snapshot.orders) == 1
        assert snapshot.orders[0].instrument_id == "BTCUSDT-PERP.BINANCE"
        assert snapshot.orders[0].remaining_quantity == Decimal("1.5")

        async def empty():
            return []

        reader.query_open_orders = empty
        assert reader.read().orders == ()
        assert reader.read().revision == 3
        print("P4-BN1a通过：成功的全市场HTTP查询可生成单调活动订单快照")

        async def failed():
            raise ConnectionError("HTTP请求失败")

        reader.query_open_orders = failed
        try:
            reader.read()
        except ConnectionError:
            pass
        else:
            raise AssertionError("HTTP失败不能伪装为空订单快照")
        assert reader._revision == 3

        async def incomplete():
            return [{"symbol": "BTCUSDT", "clientOrderId": "O-2"}]

        reader.query_open_orders = incomplete
        try:
            reader.read()
        except ValueError:
            pass
        else:
            raise AssertionError("字段不全不能开放活动订单恢复")
        assert reader._revision == 3
        print("P4-BN1b通过：HTTP失败及缺字段不会变成权威空快照或推进版本")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=3)
        loop.close()


if __name__ == "__main__":
    main()
