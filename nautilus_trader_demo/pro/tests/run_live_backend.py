"""阶段E9：Binance Stream与Nautilus Live Backend分步验证。

全部测试使用内存节点或自动成交Driver，不连接真实交易账户、不发送真实订单。
"""

from __future__ import annotations

import argparse
import threading
import time
from decimal import Decimal

from market.basic.base import DataType, InstrumentId, InstrumentMeta, QuoteTick
from market.stream.bn import BNWSStreamDataFeed
from trader import (
    BackendExecutionClient,
    DataBinding,
    ExecutionReport,
    ExecutionReportType,
    ExecutionRequest,
    ExecutionRoute,
    LiveExecutionBackendPort,
    NautilusLiveExecutionBackend,
    NautilusTradingNodeDriver,
    NetTargetOrderPlanner,
    OrderIntent,
    OrderSide,
    PositionEffect,
    PositionManager,
    RuntimeMode,
    StrategyTemplate,
    UnifiedStrategyRunner,
)


BTC_ID = InstrumentId.from_str("BTCUSDT.BINANCE")


def test1_net_target_planner() -> None:
    """E9a：目标跨零时必须明确拆成平仓和反向开仓。"""

    positions = PositionManager()
    positions.set_account_position("binance-live", BTC_ID, 2)
    planner = NetTargetOrderPlanner(positions)
    request = ExecutionRequest(
        strategy_id="e9-alpha",
        revision=1,
        client_id="binance-live",
        ts_event=1,
        targets={BTC_ID: Decimal(-1)},
        execution_policy="DIRECT",
    )
    orders = planner.plan(request)
    assert len(orders) == 2
    assert orders[0].side is OrderSide.SELL
    assert orders[0].quantity == 2
    assert orders[0].position_effect is PositionEffect.CLOSE
    assert orders[0].reduce_only is True
    assert orders[1].side is OrderSide.SELL
    assert orders[1].quantity == 1
    assert orders[1].position_effect is PositionEffect.OPEN
    assert orders[1].reduce_only is False
    print("E9a通过：净目标跨零已拆分为reduce-only平仓和反向开仓")


class _TraderProbe:
    def __init__(self) -> None:
        self.strategies = []

    def add_strategy(self, strategy) -> None:
        self.strategies.append(strategy)


class _NodeProbe:
    def __init__(self) -> None:
        self.trader = _TraderProbe()
        self.calls: list[str] = []
        self._stop = threading.Event()

    def build(self) -> None:
        self.calls.append("build")

    def run(self) -> None:
        self.calls.append("run")
        self._stop.wait(2.0)

    def stop(self) -> None:
        self.calls.append("stop")
        self._stop.set()

    def dispose(self) -> None:
        self.calls.append("dispose")


def test2_trading_node_driver_lifecycle() -> None:
    """E9b：Live Backend拥有TradingNode、Gateway、线程和对账生命周期。"""

    node = _NodeProbe()
    reconciliations: list[str] = []
    driver = NautilusTradingNodeDriver(
        "binance-live",
        node,
        reconcile_callback=lambda: (reconciliations.append("reconcile"), {})[1],
    )
    backend = NautilusLiveExecutionBackend("binance-live", driver)
    assert isinstance(backend, LiveExecutionBackendPort)
    backend.start()
    deadline = time.monotonic() + 2.0
    while "run" not in node.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert node.calls[:2] == ["build", "run"]
    assert len(node.trader.strategies) == 1
    backend.reconcile()
    assert reconciliations == ["reconcile"]
    backend.stop()
    backend.stop()
    assert node.calls[-2:] == ["stop", "dispose"]
    print("E9b通过：Nautilus TradingNode Driver生命周期与运行期对账边界正常")


class _AutoFillDriver:
    """E9c安全柜台替身：立即接受并成交，不连接任何交易所。"""

    driver_id = "binance-live"

    def __init__(self) -> None:
        self.sink = None
        self.orders: list[OrderIntent] = []
        self.reconciliations = 0
        self._sequence = 0

    def start(self, report_sink) -> None:
        self.sink = report_sink

    def stop(self) -> None:
        self.sink = None

    def submit_order(self, order: OrderIntent) -> None:
        if self.sink is None:
            raise RuntimeError("driver尚未启动")
        self.orders.append(order)
        self._sequence += 1
        order_id = f"E9-{self._sequence}"
        common = dict(
            backend_id=self.driver_id,
            client_order_id=order_id,
            instrument_id=order.instrument_id,
            ts_event=self._sequence,
            order_side=order.side,
            order_quantity=order.quantity,
            position_effect=order.position_effect,
            metadata={"strategy_id": order.strategy_id, "trade_id": f"{order_id}-trade"},
        )
        self.sink(ExecutionReport(
            report_type=ExecutionReportType.ACCEPTED,
            report_id=f"{order_id}-accepted", sequence=1, **common,
        ))
        self.sink(
            ExecutionReport(
                report_type=ExecutionReportType.FILLED,
                filled_quantity=order.quantity,
                fill_price=Decimal("80000"),
                report_id=f"{order_id}-filled", sequence=2,
                **common,
            ),
        )

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def reconcile(self):
        self.reconciliations += 1
        return {}


class _ManualBinanceStream(BNWSStreamDataFeed):
    """保留真实Binance报文转换和异步队列，只替换网络连接。"""

    def _start_network_client(self) -> None:
        pass

    def _stop_network_client(self) -> None:
        pass


class _FirstQuoteStrategy(StrategyTemplate):
    def __init__(self) -> None:
        super().__init__("e9-binance-quote")
        self.seen = 0

    def on_quote_tick(self, data_key: str, tick: QuoteTick) -> None:
        assert data_key == "btc_quote"
        self.seen += 1
        if self.seen == 1:
            self.set_target("position", 1, tick.ts_event)


def test3_binance_stream_to_live_backend() -> None:
    """E9c：Binance标准行情驱动策略、Planner、Live Backend及仓位同步。"""

    positions = PositionManager()
    driver = _AutoFillDriver()
    backend = NautilusLiveExecutionBackend("binance-live", driver)
    client = BackendExecutionClient(
        "binance-live",
        NetTargetOrderPlanner(positions),
        backend,
        positions,
    )
    feed = _ManualBinanceStream(source_id="E9_BINANCE_STREAM")
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=BTC_ID,
            price_precision=2,
            size_precision=4,
            price_increment=Decimal("0.01"),
            multiplier=Decimal(1),
            currency="USDT",
            exchange="BINANCE",
        ),
    )
    strategy = _FirstQuoteStrategy()
    runner = UnifiedStrategyRunner(
        RuntimeMode.LIVE,
        position_manager=positions,
    )
    runner.add_data_feed("binance-stream", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding("btc_quote", "binance-stream", BTC_ID, DataType.QUOTE_TICK),
        ),
        execution_routes=(
            ExecutionRoute("position", "binance-live", BTC_ID),
        ),
    )
    runner.start()
    try:
        feed.on_ws_message(
            {
                "e": "bookTicker",
                "E": 1_800_000_000_000,
                "s": "BTCUSDT",
                "b": "79999.99",
                "B": "1.2500",
                "a": "80000.00",
                "A": "2.5000",
            },
        )
        deadline = time.monotonic() + 5.0
        while positions.account_position("binance-live", BTC_ID) != 1:
            if time.monotonic() >= deadline:
                raise AssertionError("Binance行情未驱动Live Backend成交")
            time.sleep(0.01)
        assert len(driver.orders) == 1
        assert driver.orders[0].side is OrderSide.BUY
        assert positions.working_quantity("binance-live", BTC_ID) == 0
        assert len(backend.reports) == 2
        assert client.is_reconciled and not client.report_errors
    finally:
        runner.stop()
    print("E9c通过：Binance标准Stream已贯通策略、Planner、Live Backend和仓位同步")


STAGES = {
    1: test1_net_target_planner,
    2: test2_trading_node_driver_lifecycle,
    3: test3_binance_stream_to_live_backend,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="E9 Binance与Nautilus Live Backend测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for function in selected.values():
        function()


if __name__ == "__main__":
    main()
