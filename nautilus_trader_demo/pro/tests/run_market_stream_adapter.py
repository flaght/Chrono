"""阶段E8：统一行情Feed到Nautilus模拟Backend的分步测试。

E8a用内存替身验证订阅、过滤、串行转发和生命周期；E8b用真实
StreamDataFeed异步队列与Nautilus BacktestEngine完成开平仓闭环。
测试不连接网络，也不会发送真实订单。
"""

from __future__ import annotations

import argparse
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal

from bomber.backtest.config import BacktestEngineConfig
from bomber.model import Venue
from bomber.model.identifiers import InstrumentId, TraderId

from market.basic.base import (
    DataType,
    InstrumentMeta,
    MarketDataFeed,
    SubscriptionRequest,
    make_bar,
    make_quote_tick,
    make_trade_tick,
)
from market.stream.base import StreamDataFeed
from trader import (
    CtpFuturesBasicProfile,
    ExecutionBackendKind,
    ExecutionReportType,
    MarketStreamBinding,
    NautilusMarketFeedAdapter,
    OrderIntent,
    OrderSide,
    PositionEffect,
)


RB_ID = InstrumentId.from_str("rb2704.SHFE")
OTHER_ID = InstrumentId.from_str("cu2704.SHFE")


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000_000)


def _meta(instrument_id: InstrumentId = RB_ID) -> InstrumentMeta:
    return InstrumentMeta(
        instrument_id=instrument_id,
        price_precision=0,
        size_precision=0,
        price_increment=Decimal(1),
        multiplier=Decimal(10),
        currency="CNY",
        exchange="SHFE",
    )


class _ManualFeed(MarketDataFeed):
    """E8a同步Feed替身，只记录生命周期和订阅。"""

    def __init__(self) -> None:
        super().__init__("E8_MANUAL")
        self.calls: list[str] = []

    def connect(self) -> None:
        if self._is_connected:
            return
        self.calls.append("connect")
        self._is_connected = True

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self.calls.append("disconnect")
        self._is_connected = False

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        self.calls.append(f"subscribe:{request.data_type.name}")

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        self.calls.append(f"unsubscribe:{request.data_type.name}")


class _BackendProbe:
    """E8a模拟Backend替身，不包含Nautilus和撮合逻辑。"""

    kind = ExecutionBackendKind.SIMULATION

    def __init__(self) -> None:
        self.backend_id = "e8-probe"
        self.events: list[object] = []
        self.calls: list[str] = []

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def process_market_event(self, event: object):
        self.events.append(event)
        return ()

    def submit_order(self, order: OrderIntent) -> None:
        raise AssertionError("E8a不应提交订单")

    def register_report_handler(self, handler) -> None:
        pass

    def cancel_strategy(self, strategy_id: str) -> None:
        pass

    def result(self):
        return None


def test1_binding_filter_and_lifecycle() -> None:
    """E8a：验证通用桥不依赖任何具体行情源或Nautilus实现。"""

    feed = _ManualFeed()
    backend = _BackendProbe()
    bindings = (
        MarketStreamBinding(RB_ID, DataType.QUOTE_TICK),
        MarketStreamBinding(RB_ID, DataType.TRADE_TICK),
        MarketStreamBinding(RB_ID, DataType.BAR, bar_spec="1-minute"),
    )
    adapter = NautilusMarketFeedAdapter("e8-contract", feed, backend, bindings)
    adapter.start()
    adapter.start()

    quote = make_quote_tick(RB_ID, 3100, 3101, 10, 11, 1, meta=_meta())
    trade = make_trade_tick(RB_ID, 3101, 2, "e8-trade", 2, meta=_meta())
    bar = make_bar(RB_ID, 3100, 3102, 3099, 3101, 100, 3, meta=_meta())
    other = make_quote_tick(OTHER_ID, 70000, 70010, 1, 1, 4, meta=_meta(OTHER_ID))
    feed._emit_quote_tick(quote)
    feed._emit_trade_tick(trade)
    feed._emit_bar(bar)
    feed._emit_quote_tick(other)

    assert backend.events == [quote, trade, bar]
    assert adapter.events_processed == 3
    assert adapter.reports_received == 0
    assert adapter.last_error is None
    assert feed.get_subscribed_instruments() == frozenset({RB_ID})
    assert backend.calls == ["start"]
    assert feed.calls == ["connect"]

    try:
        MarketStreamBinding(RB_ID, DataType.CUSTOM_BAR)
    except ValueError:
        pass
    else:
        raise AssertionError("CustomBar不能作为第二份撮合行情推进Nautilus时钟")

    adapter.stop()
    adapter.stop()
    assert backend.calls == ["start", "stop"]
    assert feed.calls == ["connect", "disconnect"]
    print("E8a通过：订阅、事件过滤、生命周期和CustomBar边界正常")


class _ManualStreamFeed(StreamDataFeed):
    """E8b使用真实StreamDataFeed队列，但以手工push代替网络客户端。"""

    def __init__(self) -> None:
        super().__init__("E8_ASYNC_STREAM", queue_size=100)
        self.network_started = False
        self.added: list[SubscriptionRequest] = []

    def _start_network_client(self) -> None:
        self.network_started = True

    def _stop_network_client(self) -> None:
        self.network_started = False

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        self.added.append(request)

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        pass

    def push(self, event: object) -> None:
        self.enqueue_event(event)


def _wait_until(predicate, message: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def test2_async_stream_to_nautilus_round_trip() -> None:
    """E8b：异步Feed队列驱动真实Nautilus撮合、成交回报和仓位归零。"""

    from trader import NautilusSimExecutionBackend

    profile = CtpFuturesBasicProfile(
        starting_balance=Decimal("1000000"),
        commission_per_contract=Decimal("1"),
        venue=Venue("SHFE"),
    )
    instrument = profile.make_instrument(
        "rb2704",
        underlying="rb",
        price_precision=0,
        price_increment=1,
        multiplier=10,
        activation_ns=_ns("2026-01-01"),
        expiration_ns=_ns("2030-01-01"),
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.08"),
    )
    backend = NautilusSimExecutionBackend(
        "e8-stream-sim",
        BacktestEngineConfig(trader_id=TraderId("E8-TESTER"), run_analysis=False),
    )
    backend.add_profile(profile)
    backend.add_instrument(instrument)
    feed = _ManualStreamFeed()
    feed.register_instrument(_meta())
    adapter = NautilusMarketFeedAdapter(
        "e8-stream-adapter",
        feed,
        backend,
        [MarketStreamBinding(RB_ID, DataType.BAR, bar_spec="1-MINUTE")],
    )
    fills = []
    backend.register_report_handler(
        lambda report: fills.append(report)
        if report.report_type is ExecutionReportType.FILLED
        else None,
    )
    bars = [
        make_bar(
            RB_ID,
            3100 + index,
            3102 + index,
            3098 + index,
            3101 + index,
            100,
            _ns("2027-01-04T01:00:00") + index * 60_000_000_000,
            meta=_meta(),
        )
        for index in range(3)
    ]

    adapter.start()
    try:
        assert feed.is_connected and feed.network_started
        assert len(feed.added) == 0  # 连接前声明订阅，不触发动态订阅钩子。

        # 第一根Bar初始化Nautilus内部时钟、Venue和保证金账户。
        feed.push(bars[0])
        _wait_until(lambda: adapter.events_processed == 1, "初始化Bar未进入Backend")

        backend.submit_order(
            OrderIntent(
                strategy_id="e8-alpha",
                backend_id=backend.backend_id,
                instrument_id=RB_ID,
                side=OrderSide.BUY,
                quantity=1,
                position_effect=PositionEffect.OPEN,
            ),
        )
        feed.push(bars[1])
        _wait_until(lambda: len(fills) == 1, "开仓订单未由流式Bar成交")

        backend.submit_order(
            OrderIntent(
                strategy_id="e8-alpha",
                backend_id=backend.backend_id,
                instrument_id=RB_ID,
                side=OrderSide.SELL,
                quantity=1,
                position_effect=PositionEffect.CLOSE,
            ),
        )
        feed.push(bars[2])
        _wait_until(lambda: len(fills) == 2, "平仓订单未由流式Bar成交")
        assert adapter.events_processed == 3
        assert adapter.reports_received >= 2
        assert adapter.last_error is None
        assert Decimal(str(backend.engine.portfolio.net_position(RB_ID))) == 0
        orders = backend.engine.cache.orders()
        assert len(orders) == 2
        assert sum(order.is_reduce_only for order in orders) == 1
    finally:
        adapter.stop()

    result = backend.result()
    assert result is not None
    assert result.total_orders == 2
    print(
        "E8b通过：StreamDataFeed异步队列经通用适配器驱动Nautilus开平仓，"
        f"events={adapter.events_processed} orders={result.total_orders}",
    )


STAGES = {
    1: test1_binding_filter_and_lifecycle,
    2: test2_async_stream_to_nautilus_round_trip,
}


def main() -> None:
    test1_binding_filter_and_lifecycle()
    test2_async_stream_to_nautilus_round_trip()

if __name__ == "__main__":
    main()
