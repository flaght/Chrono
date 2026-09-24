"""第一类单标的EMA策略的分阶段验证。

本测试全部使用内存Bar和记录型执行端，不访问网络、不读取历史文件、不真实下单。
"""
from decimal import Decimal

from bomber.indicators import ExponentialMovingAverage


from market.basic.base import (  # noqa: E402
    Bar,
    DataType,
    InstrumentId,
    InstrumentMeta,
    MarketDataFeed,
    SubscriptionRequest,
    make_bar,
)
from trader import (  # noqa: E402
    DataBinding,
    ExecutionRequest,
    ExecutionRoute,
    RuntimeMode,
    TargetPortfolio,
    UnifiedStrategyRunner,
)
from tests.strategies.single_ema.ema_strategy import (  # noqa: E402
    EmaCrossConfig,
    EmaCrossTargetStrategy,
)


RB = InstrumentId.from_str("rb2610.SHFE")
META = InstrumentMeta(
    instrument_id=RB,
    price_precision=0,
    size_precision=0,
    price_increment=Decimal(1),
    multiplier=Decimal(10),
    currency="CNY",
    exchange="SHFE",
)



class CapturingContext:
    """纯策略测试使用的目标收集器。"""

    def __init__(self) -> None:
        self.intents: list[TargetPortfolio] = []

    def submit(self, intent: TargetPortfolio) -> None:
        self.intents.append(intent)

    def position(self, target_key: str) -> Decimal:
        del target_key
        return Decimal(0)


def _bar(close: int, ts_event: int, *, single_price: bool = False) -> Bar:
    """创建确定性的一分钟Bar；默认保留OHLC区间。"""

    return make_bar(
        instrument_id=RB,
        open=close if single_price else close - 1,
        high=close if single_price else close + 1,
        low=close if single_price else close - 1,
        close=close,
        volume=10,
        ts_event=ts_event,
        meta=META,
        bar_type="1-MINUTE",
    )



class ManualBarFeed(MarketDataFeed):
    """Runner集成测试使用的内存Bar源。"""

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


class RecordingExecutionClient:
    """只保存执行请求，不撮合也不改变仓位。"""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.requests: list[ExecutionRequest] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def submit_targets(self, request: ExecutionRequest) -> None:
        if not self.started:
            raise RuntimeError("执行客户端尚未启动")
        self.requests.append(request)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

        
def _drive(strategy: EmaCrossTargetStrategy, prices: list[int], start_ts: int = 1) -> None:
    for offset, price in enumerate(prices):
        strategy._handle_event("primary_bar", _bar(price, start_ts + offset))

def test0_native_indicator() -> None:
    """N4：确认策略直接使用Bomber原生EMA，而不是本地公式替身。"""

    strategy = EmaCrossTargetStrategy("ema-native-backend")
    assert isinstance(strategy._fast, ExponentialMovingAverage)
    assert isinstance(strategy._slow, ExponentialMovingAverage)
    assert strategy.fast_ema is None
    assert strategy.slow_ema is None
    print("N4原生指标通过：bomber.indicators.ExponentialMovingAverage")


def test1_config_and_filters() -> None:
    """验证参数约束、data_key过滤和单价Bar过滤。"""

    try:
        EmaCrossConfig(fast_period=5, slow_period=5)
    except ValueError:
        pass
    else:
        raise AssertionError("fast_period >= slow_period 应被拒绝")

    context = CapturingContext()
    strategy = EmaCrossTargetStrategy("ema-filter-test")
    strategy._bind(context)
    strategy._start()
    try:
        strategy._handle_event("other_bar", _bar(100, 1))
        strategy._handle_event("primary_bar", _bar(100, 2, single_price=True))
        assert strategy.bars_seen == 1
        assert strategy.bars_used == 0
        assert context.intents == []
    finally:
        strategy._stop()
        strategy._unbind()
    print("阶段1通过：EMA参数、data_key和单价Bar过滤正常")


def test2_example01_three_five() -> None:
    """用example01的3/5参数验证预热、多头、空头和目标防重。"""

    context = CapturingContext()
    strategy = EmaCrossTargetStrategy(
        "ema-3-5",
        EmaCrossConfig(fast_period=3, slow_period=5),
    )

    strategy._bind(context)
    strategy._start()

    try:
        _drive(strategy, [1, 2, 3, 4])
        assert not strategy.is_warmed_up
        assert context.intents == []

        _drive(strategy, [5], start_ts=5)
        assert strategy.is_warmed_up
        assert context.intents[-1].targets == {"position": Decimal(1)}
        assert context.intents[-1].revision == 1

        # 延续上涨只维持同一目标，不重复增加revision。
        _drive(strategy, [6, 7], start_ts=6)
        assert len(context.intents) == 1

        # 连续下跌使快线跌破慢线，应产生第二版空头目标。
        _drive(strategy, [5, 4, 3, 2, 1], start_ts=8)
        assert context.intents[-1].targets == {"position": Decimal(-1)}
        assert context.intents[-1].revision == 2
        assert context.intents[-1].metadata["signal"] == "SHORT"

    finally:
        strategy._stop()
        strategy._unbind()
    print("阶段2通过：example01的EMA 3/5目标逻辑正常")



def test3_example02_ten_thirty() -> None:
    """用example02的10/30参数证明同一策略实现可通过配置复用。"""

    context = CapturingContext()
    strategy = EmaCrossTargetStrategy(
        "ema-10-30",
        EmaCrossConfig(fast_period=10, slow_period=30),
    )
    strategy._bind(context)
    strategy._start()
    try:
        _drive(strategy, list(range(1, 30)))
        assert context.intents == []
        _drive(strategy, [30], start_ts=30)
        assert context.intents[-1].targets == {"position": Decimal(1)}
        assert context.intents[-1].metadata["fast_period"] == 10
        assert context.intents[-1].metadata["slow_period"] == 30
    finally:
        strategy._stop()
        strategy._unbind()
    print("阶段3通过：example02的EMA 10/30可由同一策略配置实现")


def test4_runner_integration() -> None:
    """验证DataBinding和ExecutionRoute能把EMA目标送到记录型客户端。"""

    feed = ManualBarFeed(source_id="ema-memory-bars")
    feed.register_instrument(META)
    client = RecordingExecutionClient("recording-only")
    strategy = EmaCrossTargetStrategy("ema-runner")
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed("memory-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding(
                data_key="primary_bar",
                feed_id="memory-bars",
                instrument_id=RB,
                data_type=DataType.BAR,
                bar_spec="1-MINUTE",
            ),
        ),
        execution_routes=(
            ExecutionRoute(
                target_key="position",
                client_id="recording-only",
                instrument_id=RB,
            ),
        ),
    )
    runner.start()
    try:
        for timestamp, price in enumerate([1, 2, 3, 4, 5], start=1):
            feed.push(_bar(price, timestamp))
        assert len(client.requests) == 1
        request = client.requests[0]
        assert request.logical_targets == {"position": Decimal(1)}
        assert request.targets == {RB: Decimal(1)}
        assert request.client_id == "recording-only"
    finally:
        runner.stop()
    print("阶段4通过：EMA策略已贯通Runner和静态执行路由（未撮合）")


def main() -> None:
    test0_native_indicator()
    test1_config_and_filters()
    test2_example01_three_five()
    test3_example02_ten_thirty()
    test4_runner_integration()


if __name__ == "__main__":
    main()
