"""H1-H5：行情健康状态接入策略执行闸门的分阶段验证。

测试全部使用内存Feed和记录型/内存执行端，不连接网络，不会真实下单。
"""

from __future__ import annotations

import argparse
from decimal import Decimal

from market.basic.base import (
    DataType,
    InstrumentId,
    InstrumentMeta,
    MarketDataFeed,
    SubscriptionRequest,
    make_bar,
)
from market.stream.health import StreamHealthMonitor
from strategy import (
    BackendExecutionClient,
    DataBinding,
    ExecutionRequest,
    ExecutionRoute,
    MarketHealthRejected,
    MarketRecoveryError,
    NautilusLiveExecutionBackend,
    NetTargetOrderPlanner,
    PositionManager,
    PreTradeRiskManager,
    RiskLimits,
    RiskRejected,
    RuntimeMode,
    StrategyMarketState,
    StrategyTemplate,
    TargetPortfolio,
    UnifiedStrategyRunner,
)
from strategy.execution.risk import MarketReferencePriceStore


RB = InstrumentId.from_str("rb2701.SHFE")
CU = InstrumentId.from_str("cu2701.SHFE")


class _HealthFeed(MarketDataFeed):
    def __init__(self, source_id: str) -> None:
        super().__init__(source_id)
        self._health = StreamHealthMonitor(source_id)

    @property
    def health_snapshot(self):
        return self._health.snapshot

    def register_health_handler(self, handler) -> None:
        self._health.register_handler(handler)

    def connect(self) -> None:
        self._is_connected = True
        self._health.on_connected()

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._is_connected = False
        self._health.on_disconnected()

    def interrupt(self, detail: str = "测试断流") -> None:
        self._health.on_stream_interrupted(detail)

    def push(self, event) -> None:
        if self._health.on_event(event):
            self._emit_bar(event)

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        del request

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        del request


class _PassiveStrategy(StrategyTemplate):
    pass


class _RecordingClient:
    def __init__(self, client_id: str = "recording") -> None:
        self.client_id = client_id
        self.requests: list[ExecutionRequest] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def submit_targets(self, request: ExecutionRequest) -> None:
        self.requests.append(request)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id


def _bar(instrument_id: InstrumentId, ts_event: int):
    return make_bar(
        instrument_id=instrument_id,
        open=100,
        high=101,
        low=99,
        close=100,
        volume=10,
        ts_event=ts_event,
        meta=InstrumentMeta(
            instrument_id,
            price_precision=0,
            size_precision=0,
            price_increment=Decimal(1),
        ),
        bar_type="1-MINUTE",
    )


def _intent(strategy_id: str, revision: int, quantity: int) -> TargetPortfolio:
    return TargetPortfolio(
        strategy_id=strategy_id,
        revision=revision,
        ts_event=revision,
        targets={"leg": Decimal(quantity)},
    )


def _runner():
    feed_a = _HealthFeed("H_FEED_A")
    feed_b = _HealthFeed("H_FEED_B")
    client = _RecordingClient()
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed("feed-a", feed_a)
    runner.add_data_feed("feed-b", feed_b)
    runner.add_execution_client(client)
    runner.add_strategy(
        _PassiveStrategy("strategy-a"),
        data_bindings=(DataBinding("bar", "feed-a", RB, DataType.BAR, "1-MINUTE"),),
        execution_routes=(ExecutionRoute("leg", client.client_id, RB),),
    )
    runner.add_strategy(
        _PassiveStrategy("strategy-b"),
        data_bindings=(DataBinding("bar", "feed-b", CU, DataType.BAR, "1-MINUTE"),),
        execution_routes=(ExecutionRoute("leg", client.client_id, CU),),
    )
    runner.start()
    return runner, feed_a, feed_b, client


def test1_health_aggregation() -> None:
    """H1：Runner按每个策略实际依赖的Feed汇总健康状态。"""
    runner, feed_a, feed_b, _ = _runner()
    try:
        assert runner.market_health_snapshot("strategy-a").state is StrategyMarketState.DEGRADED
        assert runner.market_health_snapshot("strategy-b").state is StrategyMarketState.DEGRADED
        feed_a.push(_bar(RB, 100))
        assert runner.market_health_snapshot("strategy-a").state is StrategyMarketState.READY
        assert runner.market_health_snapshot("strategy-b").state is StrategyMarketState.DEGRADED
        feed_b.push(_bar(CU, 100))
        assert runner.market_health_snapshot("strategy-b").state is StrategyMarketState.READY
    finally:
        runner.stop()
    print("H1通过：Runner按策略依赖分别汇总Feed健康状态")


def test2_target_gate_and_isolation() -> None:
    """H2/H3/H5：异常时禁增仓、允许向零收缩且隔离其他策略。"""
    runner, feed_a, feed_b, client = _runner()
    try:
        feed_a.push(_bar(RB, 100))
        feed_b.push(_bar(CU, 100))
        runner.submit(_intent("strategy-a", 1, 2))
        runner.submit(_intent("strategy-b", 1, 2))
        assert len(client.requests) == 2

        feed_a.interrupt()
        try:
            runner.submit(_intent("strategy-a", 2, 3))
        except MarketHealthRejected:
            pass
        else:
            raise AssertionError("异常行情期间不得增加strategy-a目标")
        assert runner.target_store.get("strategy-a").revision == 1

        runner.submit(_intent("strategy-a", 2, 1))
        assert client.requests[-1].metadata["market_health_mode"] == "REDUCE_ONLY"
        try:
            runner.submit(_intent("strategy-a", 3, -1))
        except MarketHealthRejected:
            pass
        else:
            raise AssertionError("异常行情期间不得穿越零轴反向开仓")

        # feed-a故障不能阻断只依赖feed-b的strategy-b。
        runner.submit(_intent("strategy-b", 2, 3))
        assert client.requests[-1].strategy_id == "strategy-b"
        assert client.requests[-1].metadata["market_health_mode"] == "NORMAL"
    finally:
        runner.stop()
    print("H2/H3/H5通过：禁增仓、允许减风险且不同Feed策略相互隔离")


def test3_recovery_confirmation_boundary() -> None:
    """H4边界：新鲜行情恢复后仍需带审计信息的显式确认。"""
    runner, feed_a, feed_b, _ = _runner()
    try:
        feed_a.push(_bar(RB, 100))
        feed_b.push(_bar(CU, 100))
        runner.submit(_intent("strategy-a", 1, 2))
        feed_a.interrupt()
        try:
            runner.confirm_market_recovery(
                "strategy-a",
                operator="risk-on-duty",
                reason="错误地提前确认",
            )
        except MarketRecoveryError:
            pass
        else:
            raise AssertionError("行情尚未READY时不能人工确认")

        feed_a.push(_bar(RB, 101))
        assert (
            runner.market_health_snapshot("strategy-a").state
            is StrategyMarketState.AWAITING_CONFIRMATION
        )
        confirmation = runner.confirm_market_recovery(
            "strategy-a",
            operator="risk-on-duty",
            reason="已核对断流区间且新行情连续到达",
        )
        assert confirmation.feeds == frozenset({"feed-a"})
        assert runner.market_health_snapshot("strategy-a").state is StrategyMarketState.READY
        runner.submit(_intent("strategy-a", 2, 3))
    finally:
        runner.stop()
    print("H4边界通过：恢复前置条件、人工确认和审计记录正常")


class _Driver:
    driver_id = "live"

    def __init__(self) -> None:
        self.orders = []

    def start(self, report_sink) -> None:
        self.report_sink = report_sink

    def stop(self) -> None:
        pass

    def submit_order(self, order) -> None:
        self.orders.append(order)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def reconcile(self):
        return {RB: Decimal(1)}


def test4_live_actual_position_guard() -> None:
    """H3第二道防线：Live端用实际账户仓位再次验证REDUCE_ONLY。"""
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    risk = PreTradeRiskManager("live", positions, prices, default_limits=RiskLimits())
    driver = _Driver()
    client = BackendExecutionClient(
        "live",
        NetTargetOrderPlanner(positions),
        NautilusLiveExecutionBackend("live", driver),
        positions,
        risk_manager=risk,
    )
    client.start()
    try:
        unsafe = ExecutionRequest(
            strategy_id="alpha",
            revision=1,
            client_id="live",
            ts_event=100,
            targets={RB: Decimal(2)},
            execution_policy="DIRECT",
            metadata={"market_health_mode": "REDUCE_ONLY"},
        )
        try:
            client.submit_targets(unsafe)
        except RiskRejected:
            pass
        else:
            raise AssertionError("实际账户仓位为1时，降级模式不得增至2")
        assert driver.orders == []

        safe = ExecutionRequest(
            strategy_id="alpha",
            revision=2,
            client_id="live",
            ts_event=101,
            targets={RB: Decimal(0)},
            execution_policy="DIRECT",
            metadata={"market_health_mode": "REDUCE_ONLY"},
        )
        client.submit_targets(safe)
        assert len(driver.orders) == 1
        assert driver.orders[0].reduce_only
    finally:
        client.stop()
    print("H3实仓防线通过：Live端按权威仓位再次执行REDUCE_ONLY校验")


def test5_explicit_stop_does_not_latch_recovery() -> None:
    """正常停止是生命周期操作，不能被误判为需要人工恢复的行情事故。"""
    runner, feed_a, feed_b, _ = _runner()
    feed_a.push(_bar(RB, 100))
    feed_b.push(_bar(CU, 100))
    assert runner.market_health_snapshot("strategy-a").state is StrategyMarketState.READY
    runner.stop()
    snapshot = runner.market_health_snapshot("strategy-a")
    assert snapshot.state is StrategyMarketState.DEGRADED
    assert not snapshot.confirmation_required
    print("H生命周期边界通过：主动停机不会误触发人工恢复锁")


STAGES = {
    1: test1_health_aggregation,
    2: test2_target_gate_and_isolation,
    3: test3_recovery_confirmation_boundary,
    4: test4_live_actual_position_guard,
    5: test5_explicit_stop_does_not_latch_recovery,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="H系列行情健康执行闸门测试")
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
