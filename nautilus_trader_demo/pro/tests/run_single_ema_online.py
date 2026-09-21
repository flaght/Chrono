#!/usr/bin/env python3
"""第一类单标的EMA在线行情分阶段验证。

阶段1-3完全离线；阶段4/5连接真实行情，但统一使用RecordingExecutionClient，
不会创建订单、不会连接交易柜台。
"""

from __future__ import annotations

import argparse
import os
import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy
from market.basic.base import (
    DataType,
    InstrumentId,
    InstrumentMeta,
    MarketDataFeed,
    SubscriptionRequest,
    make_trade_tick,
)
from market.stream import TradeTickBarFeed
from market.stream.bn import BNWSConfig, BNWSStreamDataFeed
from strategy import (
    DataBinding,
    ExecutionRoute,
    RecordingExecutionClient,
    RuntimeMode,
    UnifiedStrategyRunner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class _ManualBinanceFeed(BNWSStreamDataFeed):
    def _start_network_client(self) -> None:
        pass

    def _stop_network_client(self) -> None:
        pass


class _ManualTradeFeed(MarketDataFeed):
    def connect(self) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    def push(self, event) -> None:
        self._emit_trade_tick(event)

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        del request

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        del request


def _meta(instrument_id: InstrumentId) -> InstrumentMeta:
    return InstrumentMeta(
        instrument_id=instrument_id,
        price_precision=2 if "BINANCE" in str(instrument_id) else 0,
        size_precision=6 if "BINANCE" in str(instrument_id) else 0,
        price_increment=Decimal("0.01") if "BINANCE" in str(instrument_id) else Decimal(1),
        multiplier=Decimal(1 if "BINANCE" in str(instrument_id) else 10),
        currency="USDT" if "BINANCE" in str(instrument_id) else "CNY",
        exchange="BINANCE" if "BINANCE" in str(instrument_id) else str(instrument_id).split(".")[-1],
    )


def _kline(symbol: str, minute: int, close: Decimal | int, *, closed: bool = True) -> dict:
    close = Decimal(close)
    open_time = 1_800_000_000_000 + minute * 60_000
    return {
        "e": "kline",
        "E": open_time + 59_999,
        "s": symbol,
        "k": {
            "t": open_time,
            "T": open_time + 59_999,
            "s": symbol,
            "i": "1m",
            "o": str(close - Decimal("0.5")),
            "h": str(close + Decimal("1")),
            "l": str(close - Decimal("1")),
            "c": str(close),
            "v": "10.000000",
            "x": closed,
        },
    }


def _wait_for_request(client: RecordingExecutionClient, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not client.requests:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{timeout:g}秒内EMA没有生成目标请求")
        time.sleep(0.05)


def _build_recording_runner(
    feed: MarketDataFeed,
    instrument_id: InstrumentId,
    *,
    strategy_id: str,
    fast: int,
    slow: int,
    quantity: Decimal,
):
    client = RecordingExecutionClient("recording-only")
    strategy = EmaCrossTargetStrategy(
        strategy_id,
        EmaCrossConfig(
            fast_period=fast,
            slow_period=slow,
            long_quantity=quantity,
            short_quantity=-quantity,
            skip_single_price=False,
        ),
    )
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed("live-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding(
                "primary_bar",
                "live-bars",
                instrument_id,
                DataType.BAR,
                "1-MINUTE",
            ),
        ),
        execution_routes=(
            ExecutionRoute("position", client.client_id, instrument_id),
        ),
    )
    return runner, strategy, client


def test1_binance_kline_conversion() -> None:
    """进行中Kline必须忽略，已收盘Kline转换为标准永续Bar。"""
    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    feed = _ManualBinanceFeed(BNWSConfig(market_type="futures"))
    feed.register_instrument(_meta(instrument_id))
    bars = []
    feed.register_bar_handler(bars.append)
    feed.subscribe(instrument_id, DataType.BAR, "1-MINUTE")
    feed.connect()
    try:
        # 手动Feed跳过了真实的_start_network_client；该方法才会把声明式
        # _subscriptions恢复为_active_streams。离线测试直接检查订阅映射，
        # 避免把网络启动的副作用误当成Kline转换的前置条件。
        configured_streams = {
            feed._stream_name(request)
            for requests in feed._subscriptions.values()
            for request in requests
        }
        assert configured_streams == {"btcusdt@kline_1m"}
        feed.on_ws_message(_kline("BTCUSDT", 0, 80_000, closed=False))
        time.sleep(0.1)
        assert bars == []
        feed.on_ws_message(_kline("BTCUSDT", 0, 80_001, closed=True))
        deadline = time.monotonic() + 2.0
        while not bars and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(bars) == 1
        assert str(bars[0].bar_type.instrument_id) == "BTCUSDT-PERP.BINANCE"
        assert "1-MINUTE" in str(bars[0].bar_type)
        assert bars[0].close.as_decimal() == Decimal("80001.00")
    finally:
        feed.disconnect()
    print("EMA-L1通过：Binance已收盘Kline转换为标准永续Bar")


def test2_binance_kline_to_recording() -> None:
    """Binance标准Bar驱动EMA并形成Recording执行请求。"""
    instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
    feed = _ManualBinanceFeed(BNWSConfig(market_type="futures"))
    feed.register_instrument(_meta(instrument_id))
    runner, strategy, client = _build_recording_runner(
        feed,
        instrument_id,
        strategy_id="ema-binance-recording",
        fast=2,
        slow=3,
        quantity=Decimal("0.001"),
    )
    runner.start()
    try:
        for minute, close in enumerate((80_000, 80_010, 80_020)):
            feed.on_ws_message(_kline("BTCUSDT", minute, close))
        _wait_for_request(client, 3.0)
        request = client.requests[-1]
        assert request.targets == {instrument_id: Decimal("0.001")}
        assert request.metadata["signal"] == "LONG"
        assert strategy.bars_used == 3
    finally:
        runner.stop()
    print("EMA-L2通过：Binance Kline已驱动EMA进入Recording执行边界")


def test3_ctp_tick_aggregation_to_recording() -> None:
    """原生CTP同形TradeTick经通用一分钟聚合器驱动同一EMA。"""
    instrument_id = InstrumentId.from_str("rb2701.SHFE")
    upstream = _ManualTradeFeed("CTP_TRADE_PROBE")
    feed = TradeTickBarFeed("CTP_1M_BAR_PROBE", upstream)
    meta = _meta(instrument_id)
    feed.register_instrument(meta)
    bars = []
    feed.register_bar_handler(bars.append)
    runner, strategy, client = _build_recording_runner(
        feed,
        instrument_id,
        strategy_id="ema-ctp-recording",
        fast=2,
        slow=3,
        quantity=Decimal(1),
    )
    runner.start()
    try:
        base = 1_800_000_000_000_000_000
        for minute, price in enumerate((3100, 3101, 3102, 3103)):
            upstream.push(
                make_trade_tick(
                    instrument_id=instrument_id,
                    price=price,
                    size=1,
                    trade_id=f"ctp-{minute}",
                    ts_event=base + minute * 60_000_000_000,
                    meta=meta,
                ),
            )
        _wait_for_request(client, 1.0)
        assert client.requests[-1].targets == {instrument_id: Decimal(1)}
        assert strategy.bars_used == 3
        assert len(bars) == 3
        assert bars[0].close.as_decimal() == Decimal(3100)
        assert bars[0].volume.as_decimal() == Decimal(1)
    finally:
        runner.stop()
    print("EMA-L3通过：CTP TradeTick聚合分钟Bar后已驱动EMA Recording链")


def test4_binance_real_kline_to_recording() -> None:
    """连接真实Binance Kline；只记录目标，不装配交易客户端。"""
    market_type = os.getenv("BN_MARKET_TYPE", "futures").strip().lower()
    symbol = os.getenv("BN_SYMBOL", "BTCUSDT").upper()
    suffix = "-PERP" if market_type == "futures" else ""
    instrument_id = InstrumentId.from_str(f"{symbol}{suffix}.BINANCE")
    base_url = os.getenv(
        "BN_WS_BASE_URL",
        "wss://fstream.binance.com" if market_type == "futures" else "wss://stream.binance.com:9443",
    )
    feed = BNWSStreamDataFeed(BNWSConfig(ws_base_url=base_url, market_type=market_type))
    feed.register_instrument(_meta(instrument_id))
    fast = int(os.getenv("EMA_FAST", "2"))
    slow = int(os.getenv("EMA_SLOW", "3"))
    runner, _, client = _build_recording_runner(
        feed,
        instrument_id,
        strategy_id="ema-binance-live-recording",
        fast=fast,
        slow=slow,
        quantity=Decimal(os.getenv("EMA_BN_QUANTITY", "0.001")),
    )
    timeout = float(os.getenv("BN_EMA_TIMEOUT", "300"))
    print(
        f"启动Binance EMA Recording探针: {instrument_id} fast={fast} slow={slow} "
        f"timeout={timeout:g}s（不下单）",
    )
    runner.start()
    try:
        _wait_for_request(client, timeout)
        print(f"收到EMA目标请求: {client.requests[-1]}")
    finally:
        runner.stop()
    print("EMA-L4通过：Binance真实Kline已驱动EMA Recording链")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"缺少环境变量: {name}")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def test5_ctp_real_tick_to_recording() -> None:
    """连接真实CTP MdApi并把成交Tick聚合为EMA分钟Bar；绝不下单。"""
    from market.stream.ctp import CtpLiveDataFeed, CtpMdConfig

    symbol = _required_env("CTP_SYMBOL")
    exchange = os.getenv("CTP_EXCHANGE", "SHFE").upper()
    instrument_id = InstrumentId.from_str(f"{symbol}.{exchange}")
    upstream = CtpLiveDataFeed(
        CtpMdConfig(
            front=_required_env("CTP_MD_ADDRESS"),
            broker_id=_required_env("CTP_BROKER_ID"),
            user_id=_required_env("CTP_ACCOUNT_ID"),
            password=_required_env("CTP_PASSWORD"),
            flow_path=os.getenv("CTP_MD_FLOW_PATH", "/tmp/bomber-ctp-ema"),
            production_mode=_env_bool("CTP_PRODUCTION_MODE"),
        ),
    )
    feed = TradeTickBarFeed("CTP_EMA_1M", upstream)
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=int(os.getenv("CTP_PRICE_PRECISION", "0")),
            size_precision=int(os.getenv("CTP_SIZE_PRECISION", "0")),
            price_increment=Decimal(os.getenv("CTP_PRICE_INCREMENT", "1")),
            multiplier=Decimal(os.getenv("CTP_MULTIPLIER", "10")),
            currency=os.getenv("CTP_CURRENCY", "CNY"),
            exchange=exchange,
        ),
    )
    fast = int(os.getenv("EMA_FAST", "2"))
    slow = int(os.getenv("EMA_SLOW", "3"))
    runner, _, client = _build_recording_runner(
        feed,
        instrument_id,
        strategy_id="ema-ctp-live-recording",
        fast=fast,
        slow=slow,
        quantity=Decimal(os.getenv("EMA_CTP_QUANTITY", "1")),
    )
    timeout = float(os.getenv("CTP_EMA_TIMEOUT", "300"))
    print(
        f"启动CTP EMA Recording探针: {instrument_id} fast={fast} slow={slow} "
        f"timeout={timeout:g}s（不下单）",
    )
    runner.start()
    try:
        _wait_for_request(client, timeout)
        print(f"收到EMA目标请求: {client.requests[-1]}")
    finally:
        runner.stop()
    print("EMA-L5通过：CTP真实Tick聚合分钟Bar后已驱动EMA Recording链")


STAGES = {
    1: test1_binance_kline_conversion,
    2: test2_binance_kline_to_recording,
    3: test3_ctp_tick_aggregation_to_recording,
    4: test4_binance_real_kline_to_recording,
    5: test5_ctp_real_tick_to_recording,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="第一类EMA在线行情分层测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "4", "5", "offline"), default="offline")
    args = parser.parse_args()
    selected = (1, 2, 3) if args.stage == "offline" else (int(args.stage),)
    for stage in selected:
        STAGES[stage]()


if __name__ == "__main__":
    main()
