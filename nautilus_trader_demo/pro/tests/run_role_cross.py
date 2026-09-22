"""第四类M3a：纯穿越信号与Runner积木装配，均不撮合、不下真实订单。"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal

from datahub import MinimalDataHub, ObservedClose, RoleAssignment, RolePriceStore
from examples.role_cross.role_signal import RoleCrossSignal


CONTRACTS = {
    "main": "rb2609",
    "secondary": "rb2610",
    "near": "rb2611",
    "far": "rb2612",
}
PRICES = (
    (100, (100, 110, 90, 100)),  # 首帧只预热，价差为0。
    (200, (105, 100, 90, 100)),  # main上穿四角色均价。
    (300, (90, 100, 100, 100)),  # main下穿四角色均价。
    (400, (88, 100, 100, 100)),  # 保持空信号，不重复发目标。
)


def _hub() -> MinimalDataHub:
    day = date(2026, 7, 2)
    assignment = RoleAssignment(
        day, date(2026, 7, 1), 100, 100, CONTRACTS,
    )
    closes = tuple(
        ObservedClose(contract, day, timestamp, Decimal(price))
        for timestamp, prices in PRICES
        for contract, price in zip(CONTRACTS.values(), prices)
    )
    return MinimalDataHub(RolePriceStore((assignment,), closes))


def test1_signal() -> None:
    """只读研究价预热；上穿、下穿各一次；重复快照不能重发。"""
    hub = _hub()
    signal = RoleCrossSignal()
    assert signal.update(hub.snapshot(100)) is None
    up = signal.update(hub.snapshot(200))
    assert up is not None and up.direction == 1 and up.main_instrument == "rb2609"
    assert signal.update(hub.snapshot(200)) is None
    down = signal.update(hub.snapshot(300))
    assert down is not None and down.direction == -1
    assert signal.update(hub.snapshot(400)) is None
    assert signal.first_spread == 0
    assert signal.min_spread < 0 < signal.max_spread
    assert (signal.positive_frames, signal.negative_frames, signal.zero_frames) == (1, 2, 1)
    print("M3a-1通过：四角色因果复权价预热、上穿、下穿和幂等正常")


def test2_runner() -> None:
    """四张真实合约Bar到齐才决策；路由只记录逻辑rb_main目标。"""
    from examples.role_cross.role_strategy import RoleCrossTargetStrategy
    from market.basic.base import (
        Bar, DataType, InstrumentId, InstrumentMeta, MarketDataFeed,
        SubscriptionRequest, make_bar,
    )
    from strategy import (
        DataBinding, ExecutionRoute, RecordingExecutionClient,
        RuntimeMode, UnifiedStrategyRunner,
    )

    class ManualBarFeed(MarketDataFeed):
        def connect(self) -> None:
            self._is_connected = True

        def disconnect(self) -> None:
            self._is_connected = False

        def push(self, bar: Bar) -> None:
            self._emit_bar(bar)

        def _on_subscription_added(self, request: SubscriptionRequest) -> None:
            del request

        def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
            del request

    instrument_ids = tuple(
        InstrumentId.from_str(f"{symbol}.SHFE") for symbol in CONTRACTS.values()
    )
    metas = {
        item: InstrumentMeta(
            item, price_precision=0, size_precision=0,
            price_increment=Decimal(1), multiplier=Decimal(10), exchange="SHFE",
        )
        for item in instrument_ids
    }

    def bar(item: InstrumentId, timestamp: int, close: int) -> Bar:
        return make_bar(
            item, close, close + 1, close - 1, close, 1, timestamp,
            meta=metas[item], bar_type="1-MINUTE",
        )

    feed = ManualBarFeed("ROLE_CROSS_MANUAL")
    client = RecordingExecutionClient("recording-only")
    strategy = RoleCrossTargetStrategy("role-cross-test", _hub())
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL)
    runner.add_data_feed("role-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "role-bars", item, DataType.BAR, "1-MINUTE")
            for item in instrument_ids
        ),
        execution_routes=(
            ExecutionRoute("rb_main", client.client_id, instrument_ids[0]),
        ),
    )
    runner.start()
    try:
        for timestamp, prices in PRICES:
            for item, price in zip(instrument_ids[:-1], prices[:-1]):
                feed.push(bar(item, timestamp, price))
            assert strategy.complete_frames == PRICES.index((timestamp, prices))
            feed.push(bar(instrument_ids[-1], timestamp, prices[-1]))
        assert strategy.complete_frames == len(PRICES)
        assert [event.direction for event in strategy.signal_events] == [1, -1]
        assert len(client.requests) == 2
        assert [request.targets[instrument_ids[0]] for request in client.requests] == [
            Decimal(1), Decimal(-1),
        ]
        feed.push(bar(instrument_ids[0], 400, 88))
        assert len(client.requests) == 2
    finally:
        runner.stop()
    print("M3a-2通过：四路Bar同帧门控、逻辑目标和记录型执行路由正常（未撮合）")


def test3_dynamic_roll() -> None:
    """M4a：信号保持不变时，动态路由仍须先平旧主力再开新主力。"""
    from examples.role_cross.role_strategy import RoleCrossTargetStrategy
    from market.basic.base import (
        Bar, DataType, InstrumentId, InstrumentMeta, MarketDataFeed,
        SubscriptionRequest, make_bar,
    )
    from strategy import (
        ContractAssignment, DataBinding, DynamicExecutionRoute,
        PositionManager, RecordingExecutionClient, RuntimeMode,
        ScheduledContractResolver, UnifiedStrategyRunner,
    )

    class ManualBarFeed(MarketDataFeed):
        def connect(self) -> None:
            self._is_connected = True

        def disconnect(self) -> None:
            self._is_connected = False

        def push(self, bar: Bar) -> None:
            self._emit_bar(bar)

        def _on_subscription_added(self, request: SubscriptionRequest) -> None:
            del request

        def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
            del request

    class RecordingRollClient(RecordingExecutionClient):
        def __init__(self) -> None:
            super().__init__("role-roll-recording")
            self.cancels: list[str] = []

        def cancel_strategy(self, strategy_id: str) -> None:
            self.cancels.append(strategy_id)

    old_day = date(2026, 7, 27)
    new_day = date(2026, 7, 28)
    old_main = InstrumentId.from_str("rb2609.SHFE")
    new_main = InstrumentId.from_str("rb2610.SHFE")
    assignments = (
        RoleAssignment(old_day, date(2026, 7, 24), 100, 100, CONTRACTS),
        RoleAssignment(new_day, old_day, 300, 300, {
            "main": "rb2610", "secondary": "rb2609",
            "near": "rb2611", "far": "rb2612",
        }),
    )
    closes = tuple(
        ObservedClose(symbol, day, timestamp, Decimal(price))
        for day, timestamp, prices in (
            (old_day, 100, (100, 110, 90, 100)),
            (old_day, 200, (105, 100, 90, 100)),
            (old_day, 250, (105, 100, 90, 100)),
            (new_day, 300, (105, 100, 90, 100)),
        )
        for symbol, price in zip(CONTRACTS.values(), prices)
    )
    hub = MinimalDataHub(RolePriceStore(assignments, closes))
    resolver = ScheduledContractResolver((
        ContractAssignment("rb_main", old_main, 100, 100, 1),
        ContractAssignment("rb_main", new_main, 300, 300, 2),
    ))
    ids = tuple(InstrumentId.from_str(f"{symbol}.SHFE") for symbol in CONTRACTS.values())
    metas = {
        item: InstrumentMeta(
            item, price_precision=0, size_precision=0,
            price_increment=Decimal(1), multiplier=Decimal(10), exchange="SHFE",
        )
        for item in ids
    }

    def push_frame(feed: ManualBarFeed, timestamp: int, prices: tuple[int, ...]) -> None:
        for item, price in zip(ids, prices):
            feed.push(make_bar(
                item, price, price + 1, price - 1, price, 1, timestamp,
                meta=metas[item], bar_type="1-MINUTE",
            ))

    feed = ManualBarFeed("ROLE_ROLL_BARS")
    client = RecordingRollClient()
    positions = PositionManager()
    strategy = RoleCrossTargetStrategy("role-roll-test", hub)
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_data_feed("role-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "role-bars", item, DataType.BAR, "1-MINUTE")
            for item in ids
        ),
        execution_routes=(
            DynamicExecutionRoute("rb_main", client.client_id, resolver),
        ),
    )
    runner.start()
    try:
        push_frame(feed, 100, (100, 110, 90, 100))
        push_frame(feed, 200, (105, 100, 90, 100))
        assert len(strategy.signal_events) == 1
        assert client.requests[-1].targets[old_main] == 1
        positions.set_account_position(client.client_id, old_main, 1)
        push_frame(feed, 300, (105, 100, 90, 100))
        assert client.cancels == [strategy.strategy_id]
        assert any(request.targets.get(old_main) == 0 for request in client.requests)
        assert all(request.targets.get(new_main, Decimal(0)) == 0 for request in client.requests)
        assert len(strategy.signal_events) == 1  # 没有新穿越，目标仍应迁往新主力。
        positions.set_account_position(client.client_id, old_main, 0)
        runner.refresh_dynamic_routes(301)
        assert client.requests[-1].targets[old_main] == 0
        assert client.requests[-1].targets[new_main] == 1
        assert runner.target_store.get(strategy.strategy_id).targets["rb_main"] == 1
    finally:
        runner.stop()
    print("M4a通过：四角色策略逻辑目标跨换月保持，旧仓归零后才路由新主力（未撮合）")


def test4_simulation_backend() -> None:
    """M4b：合成四角色Bar验证同一策略进入原生模拟订单和成交。"""
    from bomber.backtest.config import BacktestEngineConfig
    from bomber.model import Venue
    from bomber.model.identifiers import TraderId

    from examples.role_cross.role_strategy import RoleCrossTargetStrategy
    from market.basic.base import (
        Bar, DataType, InstrumentId, InstrumentMeta, MarketDataFeed,
        SubscriptionRequest, make_bar,
    )
    from strategy import (
        ContractAssignment, CtpFuturesBasicProfile, DataBinding,
        DynamicExecutionRoute, MarketReferencePriceStore, MarketStreamBinding,
        NautilusMarketFeedAdapter, NautilusSimExecutionBackend,
        NetTargetOrderPlanner, PositionManager, PreTradeRiskManager,
        RiskLimits, RuntimeMode, ScheduledContractResolver,
        SimulationExecutionClient, UnifiedHistoricalRuntime,
        UnifiedStrategyRunner,
    )

    class ReplayFeed(MarketDataFeed):
        def __init__(self, bars: tuple[Bar, ...]) -> None:
            super().__init__("ROLE_FORMAL_SYNTHETIC")
            self.bars = bars

        def connect(self) -> None:
            self._is_connected = True

        def disconnect(self) -> None:
            self._is_connected = False

        def replay(self) -> int:
            for bar in self.bars:
                self._emit_bar(bar)
            return len(self.bars)

        def _on_subscription_added(self, request: SubscriptionRequest) -> None:
            del request

        def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
            del request

    start_ns = 1_799_024_400_000_000_000  # 2027-01-04 09:00 Asia/Shanghai
    day = date(2027, 1, 4)
    frames = (
        (100, 110, 90, 100),
        (105, 100, 90, 100),
        (106, 101, 90, 100),
        (90, 100, 100, 100),
        (89, 100, 100, 100),
    )
    ids = tuple(InstrumentId.from_str(f"{symbol}.SHFE") for symbol in CONTRACTS.values())
    closes = tuple(
        ObservedClose(symbol, day, start_ns + minute * 60_000_000_000, Decimal(price))
        for minute, prices in enumerate(frames)
        for symbol, price in zip(CONTRACTS.values(), prices)
    )
    hub = MinimalDataHub(RolePriceStore((
        RoleAssignment(day, date(2027, 1, 3), start_ns, start_ns, CONTRACTS),
    ), closes))
    metas = {
        item: InstrumentMeta(
            item, price_precision=0, size_precision=0,
            price_increment=Decimal(1), multiplier=Decimal(10), exchange="SHFE",
        )
        for item in ids
    }
    bars = tuple(
        make_bar(
            item, price, price + 1, price - 1, price, 100,
            start_ns + minute * 60_000_000_000,
            meta=metas[item], bar_type="1-MINUTE",
        )
        for minute, prices in enumerate(frames)
        for item, price in zip(ids, prices)
    )
    feed = ReplayFeed(bars)
    profile = CtpFuturesBasicProfile(
        starting_balance=Decimal("100000"), venue=Venue("SHFE"),
    )
    backend = NautilusSimExecutionBackend(
        "role-formal-test-sim",
        BacktestEngineConfig(trader_id=TraderId("ROLE-M4B-TEST"), run_analysis=False),
    )
    backend.add_profile(profile)
    for item in ids:
        backend.add_instrument(profile.make_instrument(
            str(item.symbol), underlying="rb", price_precision=0,
            price_increment=Decimal(1), multiplier=Decimal(10),
            activation_ns=start_ns - 86_400_000_000_000,
            expiration_ns=start_ns + 86_400_000_000_000,
            margin_init=Decimal("0.10"), margin_maint=Decimal("0.08"),
        ))
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    client = SimulationExecutionClient(
        backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id, positions, prices,
            instrument_limits={
                item: RiskLimits(
                    max_order_quantity=Decimal(2), max_abs_position=Decimal(1),
                    max_order_notional=Decimal("10000"),
                    max_abs_position_notional=Decimal("10000"),
                    max_market_age_ns=120 * 1_000_000_000,
                    contract_multiplier=Decimal(10),
                )
                for item in ids
            },
        ),
    )
    strategy = RoleCrossTargetStrategy("role-formal-test", hub)
    resolver = ScheduledContractResolver((
        ContractAssignment("rb_main", ids[0], start_ns, start_ns, 1),
    ))
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("role-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "role-bars", item, DataType.BAR, "1-MINUTE")
            for item in ids
        ),
        execution_routes=(
            DynamicExecutionRoute("rb_main", backend.backend_id, resolver),
        ),
    )
    adapter = NautilusMarketFeedAdapter(
        "role-formal-test-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE") for item in ids),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime("role-formal-test-runtime", runner, adapter)
    try:
        runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        assert strategy.complete_frames == len(frames)
        assert [event.direction for event in strategy.signal_events] == [1, -1]
        assert not orders.empty and not fills.empty
        assert not client.report_errors
        print(f"M4b通过：四角色Bar驱动原生模拟撮合，orders={len(orders)} fills={len(fills)}")
    finally:
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="第四类策略M3a分阶段测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "4", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("1", "all"):
        test1_signal()
    if args.stage in ("2", "all"):
        test2_runner()
    if args.stage in ("3", "all"):
        test3_dynamic_roll()
    if args.stage in ("4", "all"):
        test4_simulation_backend()


if __name__ == "__main__":
    main()
