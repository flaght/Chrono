"""统一策略框架的分阶段验证脚本。

本文件验证的是“标准行情如何驱动策略并形成执行请求”，不验证真实成交：

1. 基础契约：DataBinding、ExecutionRoute、TargetPortfolio；
2. 策略模板：脱离Runner验证行情回调和逻辑目标；
3. 行情切换：同一策略代码更换Feed；
4. 执行端切换：同一策略代码更换ExecutionClient；
5. 多策略和跨市场：一份BTC行情分别路由到国内期货标的；
6. DolphinDB实时行情驱动策略，但执行端只记录请求；
7. CTP离线Tick回放驱动策略，但不撮合；
10. Binance实时行情驱动策略，但不真实下单。

阶段8、9目前尚未实现，因此编号故意保留。在线阶段依赖环境变量、网络和交易
时段；RecordingExecutionClient 只证明已经形成 ExecutionRequest，不代表成交。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"缺少环境变量: {name}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


from market.basic.base import (  # noqa: E402
    Bar,
    DataType,
    InstrumentId,
    InstrumentMeta,
    MarketDataFeed,
    QuoteTick,
    SubscriptionRequest,
    TradeTick,
    make_bar,
)
from strategy import (  # noqa: E402
    DataBinding,
    ExecutionRequest,
    ExecutionRoute,
    RuntimeMode,
    StrategyTemplate,
    TargetPortfolio,
    UnifiedStrategyRunner,
)


BTC_ID = InstrumentId.from_str("BTCUSDT.BINANCE")
RB_ID = InstrumentId.from_str("rb2610.SHFE")
IF_ID = InstrumentId.from_str("IF2610.CFFEX")
CTP_TICK_PATH = Path(
    "/workspace/data/fut_tick/7050707549_-/2026/202607/20260701/"
    "rb2610_20260701.csv",
)


class ManualBarFeed(MarketDataFeed):
    """可由测试主动push Bar的最小行情源。

    它不访问网络和文件，用来隔离验证Runner的订阅、分发和路由逻辑。
    """

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
    """只记录ExecutionRequest的安全测试客户端。

    它实现统一执行端口，但不计算订单差额、不撮合、不改变仓位、也不访问任何
    交易柜台。因此测试通过只说明策略目标成功到达执行边界。
    """

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.requests: list[ExecutionRequest] = []
        self.started = False
        self._request_received = threading.Event()

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def submit_targets(self, request: ExecutionRequest) -> None:
        if not self.started:
            raise RuntimeError("交易客户端尚未启动")
        self.requests.append(request)
        self._request_received.set()

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def wait_for_request(self, timeout: float) -> bool:
        return self._request_received.wait(timeout)


class BtcSignalStrategy(StrategyTemplate):
    """BTC收盘价达到阈值后输出逻辑目标仓位。

    策略只认识 ``btc_signal`` 和 ``trade_leg``，不知道具体Feed、交易客户端或
    最终交易合约，是验证“策略无感知切换”的最小样例。
    """

    def __init__(self, strategy_id: str, threshold: float = 50_000) -> None:
        super().__init__(strategy_id)
        self.threshold = threshold
        self.received_data_keys: list[str] = []

    def on_bar(self, data_key: str, bar: Bar) -> None:
        self.received_data_keys.append(data_key)
        if data_key != "btc_signal":
            return
        quantity = 1 if float(bar.close) >= self.threshold else 0
        self.set_target(
            "trade_leg",
            quantity,
            bar.ts_event,
            metadata={"signal_close": str(bar.close)},
        )


class FirstQuoteTargetStrategy(StrategyTemplate):
    """收到第一份有效报价后只产生一次目标。

    ``submitted`` 防止实时行情持续到达时重复提交，适合验证一条在线链路是否
    已从Feed贯通到ExecutionRequest。
    """

    def __init__(self, strategy_id: str) -> None:
        super().__init__(strategy_id)
        self.quote_count = 0
        self.submitted = False

    def on_quote_tick(self, data_key: str, tick: QuoteTick) -> None:
        if data_key != "signal_quote":
            return
        self.quote_count += 1
        if self.submitted:
            return
        if tick.bid_price.as_double() <= 0 or tick.ask_price.as_double() <= 0:
            return
        self.submitted = True
        self.set_target(
            "trade_leg",
            1,
            tick.ts_event,
            metadata={
                "signal_bid": str(tick.bid_price),
                "signal_ask": str(tick.ask_price),
            },
        )


class CtpTickProbeStrategy(StrategyTemplate):
    """同时统计CTP Quote/Trade，并由第一份有效Quote产生一次目标。

    Quote用于触发目标，Trade只计数，从而验证同一个离线Tick源产生的两种标准
    事件都能按各自data_key送达策略。
    """

    def __init__(self, strategy_id: str) -> None:
        super().__init__(strategy_id)
        self.quote_count = 0
        self.trade_count = 0
        self.submitted = False

    def on_quote_tick(self, data_key: str, tick: QuoteTick) -> None:
        if data_key != "ctp_quote":
            return
        self.quote_count += 1
        if self.submitted:
            return
        if tick.bid_price.as_double() <= 0 or tick.ask_price.as_double() <= 0:
            return
        self.submitted = True
        self.set_target(
            "trade_leg",
            1,
            tick.ts_event,
            metadata={
                "source": "ctp-tick-file",
                "signal_bid": str(tick.bid_price),
                "signal_ask": str(tick.ask_price),
            },
        )

    def on_trade_tick(self, data_key: str, tick: TradeTick) -> None:
        del tick
        if data_key == "ctp_trade":
            self.trade_count += 1


class CapturingContext:
    """单独测试StrategyTemplate的最小上下文替身。

    submit()只保存TargetPortfolio，position()固定返回0。它不包含Feed、Runner、
    交易客户端和真实仓位，用于证明策略逻辑本身可以独立测试。
    """

    def __init__(self) -> None:
        self.intents: list[TargetPortfolio] = []

    def submit(self, intent: TargetPortfolio) -> None:
        self.intents.append(intent)

    def position(self, target_key: str) -> Decimal:
        del target_key
        return Decimal(0)


def test1_contracts() -> None:
    """第一阶段：验证三个基础契约的字段语义和标准化。

    DataBinding把具体行情映射为策略逻辑键；ExecutionRoute把逻辑目标映射到客户
    端和真实合约；TargetPortfolio表达策略希望最终持有的数量。这里不启动Runner。
    """

    # BTC的一分钟Bar在策略内部统一称为btc_signal。
    binding = DataBinding(
        data_key="btc_signal",
        feed_id="file-replay",
        instrument_id=BTC_ID,
        data_type=DataType.BAR,
        bar_spec="1-minute",
    )
    # trade_leg最终可以跨市场路由到Bomber中的rb2610，而不是必须交易BTC。
    route = ExecutionRoute(
        target_key="trade_leg",
        client_id="bomber-ctp",
        instrument_id=RB_ID,
    )
    intent = TargetPortfolio(
        strategy_id="contract-test",
        revision=1,
        ts_event=1,
        targets={"trade_leg": 2},
    )
    # 验证bar_spec统一大写、InstrumentId保持不变、数量规范成Decimal。
    assert binding.bar_spec == "1-MINUTE"
    assert route.instrument_id == RB_ID
    assert intent.targets == {"trade_leg": Decimal(2)}
    print("阶段1通过：数据绑定、执行路由和目标组合契约正常")


def test2_template() -> None:
    """第二阶段：脱离Runner验证策略的纯信号逻辑。

    手动绑定CapturingContext并推送一根BTC Bar，验证策略收到逻辑data_key、生成
    一个目标且revision从1开始。通过不代表行情订阅或执行路由已经工作。
    """

    context = CapturingContext()
    strategy = BtcSignalStrategy("template-test")
    strategy._bind(context)
    strategy._start()
    try:
        # 60050高于默认阈值50000，因此策略应产生trade_leg=+1。
        strategy._handle_event("btc_signal", _btc_bar())
        assert strategy.received_data_keys == ["btc_signal"]
        assert len(context.intents) == 1
        assert context.intents[0].targets == {"trade_leg": Decimal(1)}
        assert context.intents[0].revision == 1
    finally:
        strategy._stop()
        strategy._unbind()
    print("阶段2通过：策略模板收到逻辑数据并产生逻辑目标")


def test3_data_source_switch() -> None:
    """第三阶段：验证策略不感知行情源名称。

    三个feed_id都使用ManualBarFeed模拟，但装配名称分别代表文件、CTP和DolphinDB。
    相同策略类应产生完全相同的逻辑目标。该测试验证绑定机制，不连接真实数据源。
    """

    for feed_id in ("file-replay", "ctp-live", "dolphin-live"):
        request = _run_single_case(feed_id=feed_id, client_id="nt-sim")
        assert request.logical_targets == {"trade_leg": Decimal(1)}
    print("阶段3通过：策略不感知文件、CTP、DolphinDB行情源切换")


def test4_execution_client_switch() -> None:
    """第四阶段：验证策略不感知执行客户端。

    只改变client_id，策略及行情不变。Runner必须把相同逻辑目标路由到指定客户端
    和rb2610。这里三个客户端都是RecordingExecutionClient替身，不是真实适配器。
    """

    for client_id in ("nt-sim", "bomber-ctp", "vnpy-ctp"):
        request = _run_single_case(feed_id="file-replay", client_id=client_id)
        assert request.client_id == client_id
        assert request.targets == {RB_ID: Decimal(1)}
    print("阶段4通过：策略不感知NT、Bomber、vn.py交易客户端切换")


def test5_multi_strategy_cross_market() -> None:
    """第五阶段：验证共享行情、多策略分发和跨市场静态路由。

    同一根BTC Bar同时驱动两个策略：一个把trade_leg路由到Bomber的RB，另一个
    路由到vn.py的IF。通过说明行情标的与交易标的可以不同，但尚未进行多策略净额。
    """

    feed = _manual_feed("binance-live")
    bomber = RecordingExecutionClient("bomber-ctp")
    vnpy = RecordingExecutionClient("vnpy-ctp")
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed("binance-live", feed)
    runner.add_execution_client(bomber)
    runner.add_execution_client(vnpy)
    _add_strategy(
        runner,
        BtcSignalStrategy("btc-to-rb"),
        feed_id="binance-live",
        client_id="bomber-ctp",
        execution_instrument=RB_ID,
    )
    _add_strategy(
        runner,
        BtcSignalStrategy("btc-to-if"),
        feed_id="binance-live",
        client_id="vnpy-ctp",
        execution_instrument=IF_ID,
    )
    runner.start()
    try:
        # 一次发布应同时命中两个注册策略，各自产生一份ExecutionRequest。
        feed.push(_btc_bar())
        assert bomber.requests[0].targets == {RB_ID: Decimal(1)}
        assert vnpy.requests[0].targets == {IF_ID: Decimal(1)}
    finally:
        runner.stop()
    print("阶段5通过：同一BTC行情驱动多个策略，并路由到不同CTP标的和客户端")


def test6_dolphin_live_to_recording_execution() -> None:
    """第六阶段：验证真实DolphinDB流表到策略执行边界的完整在线链路。

    路径为：DolphinDB流表行 → Converter → QuoteTick → Runner/DataBinding →
    FirstQuoteTargetStrategy → TargetPortfolio → ExecutionRoute → ExecutionRequest。
    RecordingExecutionClient确保本测试绝不会真实下单。
    """
    from market.stream.dolphin import (
        DolphinDbConfig,
        DolphinDbLiveDataFeed,
        DolphinDbStreamSpec,
    )

    symbol = _required_env("DDB_SYMBOL").strip()
    exchange = os.getenv("DDB_EXCHANGE", "SHFE").upper()
    instrument_id = InstrumentId.from_str(f"{symbol}.{exchange}")
    timeout = float(os.getenv("STRATEGY_LIVE_TIMEOUT", "60"))
    feed_id = "dolphin-live"
    client_id = "recording-only"
    # offset/resub/batch/throttle只属于数据源装配，策略不会看到这些配置。
    stream = DolphinDbStreamSpec.tick(
        table_name=_required_env("DDB_TICK_STREAM_TABLE"),
        action_name=os.getenv("DDB_STRATEGY_ACTION", "bomberStrategyCuProbe"),
        offset=int(os.getenv("DDB_STREAM_OFFSET", "-1")),
        resub=_env_bool("DDB_STREAM_RESUB", True),
        batch_size=int(os.getenv("DDB_STREAM_BATCH_SIZE", "0")),
        throttle=float(os.getenv("DDB_STREAM_THROTTLE", "0.01")),
    )
    feed = DolphinDbLiveDataFeed(
        DolphinDbConfig(
            host=_required_env("DDB_HOST"),
            port=int(os.getenv("DDB_PORT", "8848")),
            username=_required_env("DDB_USERNAME"),
            password=_required_env("DDB_PASSWORD"),
            streams=(stream,),
            streaming_port=int(os.getenv("DDB_STREAMING_PORT", "0")),
            keep_alive_seconds=int(os.getenv("DDB_KEEP_ALIVE_SECONDS", "60")),
        ),
    )
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=int(os.getenv("DDB_PRICE_PRECISION", "0")),
            size_precision=int(os.getenv("DDB_SIZE_PRECISION", "0")),
            price_increment=Decimal(os.getenv("DDB_PRICE_INCREMENT", "10")),
            multiplier=Decimal(os.getenv("DDB_MULTIPLIER", "5")),
            currency=os.getenv("DDB_CURRENCY", "CNY"),
            exchange=exchange,
        ),
    )
    client = RecordingExecutionClient(client_id)
    strategy = FirstQuoteTargetStrategy("dolphin-cu-probe")
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed(feed_id, feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding(
                data_key="signal_quote",
                feed_id=feed_id,
                instrument_id=instrument_id,
                data_type=DataType.QUOTE_TICK,
            ),
        ),
        execution_routes=(
            ExecutionRoute(
                target_key="trade_leg",
                client_id=client_id,
                instrument_id=instrument_id,
            ),
        ),
    )
    print(
        "启动策略真实行情探针："
        f"feed=DolphinDB instrument={instrument_id} execution={client_id}",
    )
    try:
        runner.start()
        if not client.wait_for_request(timeout):
            raise TimeoutError(
                f"{timeout:g}秒内策略没有产生执行请求；"
                "请确认DolphinDB流表仍在推送该合约",
            )
        # 策略有防重标记，因此持续行情只能产生第一份请求。
        assert len(client.requests) == 1
        request = client.requests[0]
        assert request.client_id == client_id
        assert request.logical_targets == {"trade_leg": Decimal(1)}
        assert request.targets == {instrument_id: Decimal(1)}
        assert strategy.quote_count >= 1
        print(
            "记录到ExecutionRequest："
            f"strategy={request.strategy_id} revision={request.revision} "
            f"client={request.client_id} targets={dict(request.targets)} "
            f"metadata={dict(request.metadata)}",
        )
        print("阶段6通过：真实DolphinDB行情已驱动策略并完成执行路由（未真实下单）")
    finally:
        runner.stop()


def test7_ctp_tick_replay_to_strategy() -> None:
    """第七阶段：验证CTP离线CSV Tick到策略的确定性回放链路。

    同一原始记录可产生QuoteTick及由累计Volume变化推导的TradeTick。测试核对Feed
    汇总计数与策略计数完全一致，并验证策略只提交一次目标。它不包含撮合和PnL。
    """
    from market.replay.base import FileReplayFeed
    from market.replay.parsers.tick import CtpTickParser

    if not CTP_TICK_PATH.is_file():
        raise FileNotFoundError(f"CTP Tick文件不存在: {CTP_TICK_PATH}")
    feed_id = "ctp-tick-replay"
    client_id = "recording-only"
    feed = FileReplayFeed(source_id="CTP_TICK_STRATEGY_REPLAY")
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=RB_ID,
            price_precision=0,
            size_precision=0,
            price_increment=Decimal("1"),
            multiplier=Decimal("10"),
            currency="CNY",
            exchange="SHFE",
        ),
    )
    feed.add_tick_csv(CTP_TICK_PATH, CtpTickParser(exchange="SHFE"))
    client = RecordingExecutionClient(client_id)
    strategy = CtpTickProbeStrategy("ctp-tick-file-probe")
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL)
    runner.add_data_feed(feed_id, feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding(
                data_key="ctp_quote",
                feed_id=feed_id,
                instrument_id=RB_ID,
                data_type=DataType.QUOTE_TICK,
            ),
            DataBinding(
                data_key="ctp_trade",
                feed_id=feed_id,
                instrument_id=RB_ID,
                data_type=DataType.TRADE_TICK,
            ),
        ),
        execution_routes=(
            ExecutionRoute(
                target_key="trade_leg",
                client_id=client_id,
                instrument_id=RB_ID,
            ),
        ),
    )
    print(
        "启动CTP Tick策略回放："
        f"file={CTP_TICK_PATH} instrument={RB_ID} execution={client_id}",
    )
    try:
        # run_replay会按事件时间顺序播放完整文件，并返回各事件类型计数。
        summary = runner.run_replay()
        assert summary.quote_ticks > 0
        assert summary.trade_ticks > 0
        # 相等说明订阅过滤和Runner分发没有丢失或重复事件。
        assert strategy.quote_count == summary.quote_ticks
        assert strategy.trade_count == summary.trade_ticks
        assert len(client.requests) == 1
        request = client.requests[0]
        assert request.client_id == client_id
        assert request.logical_targets == {"trade_leg": Decimal(1)}
        assert request.targets == {RB_ID: Decimal(1)}
        assert request.metadata["source"] == "ctp-tick-file"
        print(
            "策略收到事件："
            f"QuoteTick={strategy.quote_count:,} TradeTick={strategy.trade_count:,}",
        )
        print(
            "记录到ExecutionRequest："
            f"strategy={request.strategy_id} revision={request.revision} "
            f"client={request.client_id} targets={dict(request.targets)}",
        )
        print("阶段7通过：CTP离线Tick已驱动统一策略（未撮合、未真实下单）")
    finally:
        runner.stop()


def test10_binance_live_to_recording_execution() -> None:
    """第十阶段：验证Binance WebSocket实时报价驱动统一策略。

    路径为：bookTicker → QuoteTick → Runner → FirstQuoteTargetStrategy →
    ExecutionRequest。执行端仍是recording-only，所以测试不会向Binance发送订单。
    """
    from market.stream.bn import BNWSConfig, BNWSStreamDataFeed

    timeout = float(os.getenv("BN_STRATEGY_TIMEOUT", "30"))
    feed_id = "binance-live"
    client_id = "recording-only"
    feed = BNWSStreamDataFeed(
        BNWSConfig(
            ws_base_url=os.getenv(
                "BN_WS_BASE_URL",
                "wss://stream.binance.com:9443",
            ),
        ),
    )
    feed.register_instrument(_btc_meta())
    client = RecordingExecutionClient(client_id)
    strategy = FirstQuoteTargetStrategy("binance-btc-quote-probe")
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed(feed_id, feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding(
                data_key="signal_quote",
                feed_id=feed_id,
                instrument_id=BTC_ID,
                data_type=DataType.QUOTE_TICK,
            ),
        ),
        execution_routes=(
            ExecutionRoute(
                target_key="trade_leg",
                client_id=client_id,
                instrument_id=BTC_ID,
            ),
        ),
    )
    print(
        "启动Binance在线策略探针："
        f"instrument={BTC_ID} execution={client_id}（不真实下单）",
    )
    try:
        runner.start()
        # 等待策略目标到达执行端；超时时附带Feed保存的WebSocket错误。
        if not client.wait_for_request(timeout):
            detail = (
                f"；WebSocket错误={feed.last_ws_error}"
                if feed.last_ws_error
                else ""
            )
            raise TimeoutError(f"{timeout:g}秒内策略没有产生执行请求{detail}")
        assert len(client.requests) == 1
        request = client.requests[0]
        assert request.client_id == client_id
        assert request.logical_targets == {"trade_leg": Decimal(1)}
        assert request.targets == {BTC_ID: Decimal(1)}
        assert strategy.quote_count >= 1
        print(
            "记录到ExecutionRequest："
            f"strategy={request.strategy_id} revision={request.revision} "
            f"client={request.client_id} targets={dict(request.targets)} "
            f"metadata={dict(request.metadata)}",
        )
        print("阶段10通过：Binance实时行情已生成下单请求（未真实下单）")
    finally:
        runner.stop()


def _run_single_case(feed_id: str, client_id: str) -> ExecutionRequest:
    """装配一个最小Runner案例，供阶段3和4复用。

    只改变feed_id/client_id，其他输入保持一致，以确保测试每次只改变一个变量。
    """

    feed = _manual_feed(feed_id)
    client = RecordingExecutionClient(client_id)
    strategy = BtcSignalStrategy(f"strategy-{feed_id}-{client_id}")
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed(feed_id, feed)
    runner.add_execution_client(client)
    _add_strategy(
        runner,
        strategy,
        feed_id=feed_id,
        client_id=client_id,
        execution_instrument=RB_ID,
    )
    runner.start()
    try:
        feed.push(_btc_bar())
        assert len(client.requests) == 1
        return client.requests[0]
    finally:
        runner.stop()


def _add_strategy(
    runner: UnifiedStrategyRunner,
    strategy: StrategyTemplate,
    *,
    feed_id: str,
    client_id: str,
    execution_instrument: InstrumentId,
) -> None:
    """集中声明测试策略的数据绑定和静态执行路由。"""

    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding(
                data_key="btc_signal",
                feed_id=feed_id,
                instrument_id=BTC_ID,
                data_type=DataType.BAR,
                bar_spec="1-MINUTE",
            ),
        ),
        execution_routes=(
            ExecutionRoute(
                target_key="trade_leg",
                client_id=client_id,
                instrument_id=execution_instrument,
            ),
        ),
    )


def _manual_feed(source_id: str) -> ManualBarFeed:
    """创建注册了BTC元数据的手动行情源。"""

    feed = ManualBarFeed(source_id=source_id)
    feed.register_instrument(_btc_meta())
    return feed


def _btc_meta() -> InstrumentMeta:
    """返回测试用BTC精度和币种信息。"""

    return InstrumentMeta(
        instrument_id=BTC_ID,
        price_precision=2,
        size_precision=6,
        price_increment=Decimal("0.01"),
        exchange="BINANCE",
        currency="USDT",
    )


def _btc_bar() -> Bar:
    """创建收盘价高于策略阈值的确定性一分钟Bar。"""

    return make_bar(
        instrument_id=BTC_ID,
        open=60_000,
        high=60_100,
        low=59_900,
        close=60_050,
        volume=10,
        ts_event=1_000_000_000,
        meta=_btc_meta(),
        bar_type="1-MINUTE",
    )


STAGES: dict[int, tuple[str, Any]] = {
    1: ("基础数据契约", test1_contracts),
    2: ("策略模板", test2_template),
    3: ("行情源切换", test3_data_source_switch),
    4: ("交易客户端切换", test4_execution_client_switch),
    5: ("多策略与跨市场路由", test5_multi_strategy_cross_market),
    6: ("真实DolphinDB行情与记录型执行端", test6_dolphin_live_to_recording_execution),
    7: ("CTP离线Tick驱动策略", test7_ctp_tick_replay_to_strategy),
    10: ("Binance实时行情驱动策略", test10_binance_live_to_recording_execution),
}


def main() -> None:
    """按命令行选择阶段；all会包含需要真实网络的在线阶段。"""
    test1_contracts()
    test2_template()
    test3_data_source_switch()
    test4_execution_client_switch()
    test5_multi_strategy_cross_market()
    #test6_dolphin_live_to_recording_execution()
    test7_ctp_tick_replay_to_strategy()
    test10_binance_live_to_recording_execution()


if __name__ == "__main__":
    main()
