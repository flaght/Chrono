"""I1-I3：回测与实盘统一积木主链的离线分阶段验证。"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal

from bomber.backtest.config import BacktestEngineConfig
from bomber.model import Venue
from bomber.model.identifiers import InstrumentId, TraderId

from examples.single_ema.ema_ctp_backtest import _future
from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy
from market.basic.base import (
    DataType,
    InstrumentMeta,
    MarketDataFeed,
    SubscriptionRequest,
    make_bar,
)
from strategy import (
    CtpFuturesBasicProfile,
    DataBinding,
    ExecutionBackendKind,
    ExecutionReport,
    ExecutionReportType,
    ExecutionRequest,
    ExecutionRoute,
    MarketStreamBinding,
    MarketReferencePriceStore,
    NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend,
    NetTargetOrderPlanner,
    OrderSide,
    PositionManager,
    PreTradeRiskManager,
    RuntimeMode,
    SimulationExecutionClient,
    StrategyTemplate,
    TargetPortfolio,
    UnifiedHistoricalRuntime,
    UnifiedStrategyRunner,
)


RB = InstrumentId.from_str("rb9999.SHFE")


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000_000)


def _meta() -> InstrumentMeta:
    return InstrumentMeta(
        RB,
        price_precision=0,
        size_precision=0,
        price_increment=Decimal(1),
        multiplier=Decimal(10),
        exchange="SHFE",
    )


def _bars(count: int = 10):
    prices = [100, 101, 103, 105, 102, 99, 97, 100, 104, 106][:count]
    return tuple(
        make_bar(
            RB,
            price - 1,
            price + 1,
            price - 2,
            price,
            100,
            _ns("2027-01-04T01:00:00") + index * 60_000_000_000,
            meta=_meta(),
            bar_type="1-MINUTE",
        )
        for index, price in enumerate(prices)
    )


class _ReplayFeed(MarketDataFeed):
    def __init__(self, events) -> None:
        super().__init__("I_REPLAY")
        self.events = tuple(events)

    def connect(self) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    def replay(self):
        for event in self.events:
            self._emit_bar(event)
        return {"bars": len(self.events)}

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        del request

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        del request


class _BackendProbe:
    kind = ExecutionBackendKind.SIMULATION

    def __init__(
        self,
        backend_id: str = "sim",
        log: list[str] | None = None,
        *,
        auto_fill: bool = False,
    ) -> None:
        self.backend_id = backend_id
        self.log = log if log is not None else []
        self.orders = []
        self.handlers = []
        self.started = False
        self.auto_fill = auto_fill
        self.filled_orders = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def submit_order(self, order) -> None:
        self.orders.append(order)
        self.log.append(f"order:{order.strategy_id}")

    def register_report_handler(self, handler) -> None:
        self.handlers.append(handler)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def process_market_event(self, event):
        self.log.append(f"market:{event.ts_event}")
        reports = []
        if self.auto_fill:
            while self.filled_orders < len(self.orders):
                reports.append(self.emit_fill(self.filled_orders, event.ts_event))
                self.filled_orders += 1
        return tuple(reports)

    def result(self):
        return {"orders": len(self.orders)}

    def finish(self):
        return self.result()

    def emit_fill(self, order_index: int, ts_event: int) -> ExecutionReport:
        order = self.orders[order_index]
        report = ExecutionReport(
            backend_id=self.backend_id,
            client_order_id=f"probe-{order_index}",
            instrument_id=order.instrument_id,
            report_type=ExecutionReportType.FILLED,
            ts_event=ts_event,
            filled_quantity=order.quantity,
            fill_price=100,
            order_side=order.side,
            order_quantity=order.quantity,
            position_effect=order.position_effect,
            report_id=f"probe-fill-{order_index}",
            sequence=1,
        )
        for handler in tuple(self.handlers):
            handler(report)
        self.log.append(f"fill:{ts_event}")
        return report


def test1_simulation_execution_client() -> None:
    """I1：模拟Client复用目标、Planner、风控边界和统一回报仓位语义。"""
    positions = PositionManager()
    backend = _BackendProbe()
    client = SimulationExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id,
            positions,
            MarketReferencePriceStore(),
        ),
    )
    client.start()
    # I1故意不填参考价：默认RiskLimits不需要价格，风控时间仍应取请求时间，
    # 不能对单个int调用max并抛TypeError。
    assert client.risk_manager.price_store.get(RB) is None
    client.submit_targets(
        ExecutionRequest(
            strategy_id="alpha",
            revision=1,
            client_id=backend.backend_id,
            ts_event=100,
            targets={RB: Decimal(2)},
            execution_policy="DIRECT",
        ),
    )
    assert len(backend.orders) == 1
    assert positions.working_quantity(backend.backend_id, RB) == 2
    backend.emit_fill(0, 101)
    assert positions.account_position(backend.backend_id, RB) == 2
    assert positions.working_quantity(backend.backend_id, RB) == 0
    assert not client.report_errors
    client.stop()
    print("I1通过：SimulationExecutionClient贯通Planner、Backend、回报和仓位")


class _OneShotStrategy(StrategyTemplate):
    def __init__(self, strategy_id: str, log: list[str]) -> None:
        super().__init__(strategy_id)
        self.log = log
        self.sent = False

    def on_bar(self, data_key, bar) -> None:
        self.log.append(f"strategy:{bar.ts_event}")
        if not self.sent:
            self.sent = True
            self.set_target("position", 1, bar.ts_event)


def test2_runtime_ordering() -> None:
    """I2/I3：Runtime统一生命周期并固定先推进市场、后运行策略。"""
    log: list[str] = []
    events = _bars(2)
    feed = _ReplayFeed(events)
    positions = PositionManager()
    backend = _BackendProbe(log=log, auto_fill=True)
    client = SimulationExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id,
            positions,
            MarketReferencePriceStore(),
        ),
    )
    strategy = _OneShotStrategy("ordering", log)
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_data_feed("replay", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(DataBinding("bar", "replay", RB, DataType.BAR, "1-MINUTE"),),
        execution_routes=(ExecutionRoute("position", backend.backend_id, RB),),
    )
    adapter = NautilusMarketFeedAdapter(
        "clock",
        feed,
        backend,
        (MarketStreamBinding(RB, DataType.BAR, "1-MINUTE"),),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime("ordering-runtime", runner, adapter)
    result = runtime.run()
    assert log[:4] == [
        f"market:{events[0].ts_event}",
        f"strategy:{events[0].ts_event}",
        "order:ordering",
        f"market:{events[1].ts_event}",
    ]
    assert log[4] == f"fill:{events[1].ts_event}"
    assert positions.account_position(backend.backend_id, RB) == 1
    assert result.replay_summary == {"bars": 2}
    assert result.market_events_processed == 2
    runtime.stop()
    print("I2/I3通过：统一Historical Runtime生命周期和N→N+1事件顺序正常")


def test3_real_nautilus_pipeline() -> None:
    """I3真实引擎：StrategyTemplate不经旧Bridge也能完成目标、订单和成交。"""
    bars = _bars()
    feed = _ReplayFeed(bars)
    positions = PositionManager()
    profile = CtpFuturesBasicProfile(venue=Venue("SHFE"))
    backend = NautilusSimExecutionBackend(
        "i3-sim",
        BacktestEngineConfig(trader_id=TraderId("I3-TESTER"), run_analysis=False),
    )
    backend.add_profile(profile)
    backend.add_instrument(_future(RB, "2030-01-01"))
    client = SimulationExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id,
            positions,
            MarketReferencePriceStore(),
        ),
    )
    strategy = EmaCrossTargetStrategy(
        "i3-ema",
        EmaCrossConfig(fast_period=2, slow_period=3),
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_data_feed("replay", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding("primary_bar", "replay", RB, DataType.BAR, "1-MINUTE"),
        ),
        execution_routes=(ExecutionRoute("position", backend.backend_id, RB),),
    )
    adapter = NautilusMarketFeedAdapter(
        "i3-clock",
        feed,
        backend,
        (MarketStreamBinding(RB, DataType.BAR, "1-MINUTE"),),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime("i3-runtime", runner, adapter)
    try:
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        assert strategy.bars_used == len(bars)
        assert strategy.last_target is not None
        assert not orders.empty and not fills.empty
        assert result.backend_result.total_orders == len(orders)
        assert not client.report_errors
        print(
            "I3真实引擎通过：统一主链完成EMA目标、Nautilus订单与成交，"
            f"orders={len(orders)} fills={len(fills)}",
        )
    finally:
        runtime.stop()


STAGES = {
    1: test1_simulation_execution_client,
    2: test2_runtime_ordering,
    3: test3_real_nautilus_pipeline,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="统一正式回测主链分阶段验证")
    parser.add_argument(
        "--stage",
        choices=tuple(str(stage) for stage in STAGES) + ("all",),
        default="all",
    )
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for function in selected.values():
        function()


if __name__ == "__main__":
    main()
