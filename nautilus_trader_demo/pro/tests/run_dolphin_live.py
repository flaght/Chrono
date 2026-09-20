from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any
from decimal import Decimal

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



class FakeSession:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[Any, ...]] = []
        self.callback: Any = None

    def connect(self, host: str, port: int, username: str, password: str) -> bool:
        self.calls.append(("connect", host, port, username, password))
        return True

    def enableStreaming(self, *args: Any) -> None:
        self.calls.append(("enableStreaming", *args))

    def subscribe(self, **kwargs: Any) -> None:
        self.calls.append(("subscribe", kwargs))
        self.callback = kwargs["handler"]

    def unsubscribe(self, host: str, port: int, table: str, action: str) -> None:
        self.calls.append(("unsubscribe", host, port, table, action))

    def close(self) -> None:
        self.calls.append(("close",))


class FakeDolphinDbDriver:
    def __init__(self) -> None:
        self.start_args: dict[str, Any] = {}
        self.handlers: dict[tuple[str, str], Any] = {}
        self.started = False
        self.stopped = False

    def start(self, **kwargs: Any) -> None:
        self.start_args = kwargs
        self.started = True

    def subscribe(self, subscription: Any, handler: Any) -> None:
        self.handlers[(subscription.table_name, subscription.action_name)] = handler

    def unsubscribe(self, subscription: Any) -> None:
        self.handlers.pop((subscription.table_name, subscription.action_name), None)

    def stop(self) -> None:
        self.stopped = True

    def publish(self, spec: DolphinDbStreamSpec, row: dict[str, Any]) -> None:
        self.handlers[(spec.table_name, spec.action_name)](row)



def _tick(volume: int, last: float, millisec: int) -> dict[str, object]:
    return {
        "Code": "rb2701",
        "ActionDay": "2026-09-18",
        "UpdateTime": "10:30:00",
        "UpdateMillisec": millisec,
        "receivedTime": f"2026-09-18T10:30:00.{millisec:03d}",
        "perPenetrationTime": 125_000,
        "LastPrice": last,
        "Volume": volume,
        "BidPrice1": 3096,
        "BidVolume1": 10 + millisec,
        "AskPrice1": 3097,
        "AskVolume1": 20,
    }


def _bar() -> dict[str, object]:
    return {
        "minTime": "10:30:00.000",
        "date": "2026-09-18",
        "Code": "rb2701",
        "open": 3090,
        "high": 3100,
        "low": 3088,
        "close": 3097,
        "prev_close": 3083,
        "prev_settlement": 3085,
        "volume": 123,
        "turnover": 3_800_000,
        "pct_change": 0.0045,
        "open_interest": 1_500_000,
        "prev_open_interest": 1_490_000,
        "updateTime": "2026-09-18T10:31:00.050",
    }


def _tick_row(
    volume: int,
    millisec: int,
    last: float,
    bid_size: int,
) -> dict[str, Any]:
    return {
        "date": "2026-09-18",
        "Code": "rb2701",
        "TradingDay": "2026-09-18",
        "ActionDay": "2026-09-18",
        "tradeTime": "10:30:00.000",
        "OpenPrice": 3080.0,
        "HighestPrice": 3100.0,
        "LowestPrice": 3070.0,
        "LastPrice": last,
        "AveragePrice": 3090.0,
        "Volume": volume,
        "Turnover": 3_800_000.0,
        "BidPrice1": 3096.0,
        "BidVolume1": bid_size,
        "AskPrice1": 3097.0,
        "AskVolume1": 20,
        "PreSettlementPrice": 3085.0,
        "PreClosePrice": 3083.0,
        "PreOpenInterest": 1_490_000.0,
        "OpenInterest": 1_500_000.0,
        "UpperLimitPrice": 3300.0,
        "LowerLimitPrice": 2800.0,
        "UpdateTime": "10:30:00",
        "UpdateMillisec": millisec,
        "receivedTime": f"2026-09-18T10:30:00.{millisec:03d}",
    }


def _bar_row() -> dict[str, Any]:
    return {
        "minTime": "10:30:00.000",
        "date": "2026-09-18",
        "Code": "rb2701",
        "open": 3090.0,
        "high": 3100.0,
        "low": 3088.0,
        "close": 3097.0,
        "prev_close": 3083.0,
        "prev_settlement": 3085.0,
        "volume": 123,
        "turnover": 3_800_000.0,
        "pct_change": 0.0045,
        "open_interest": 1_500_000.0,
        "prev_open_interest": 1_490_000.0,
        "updateTime": "2026-09-18T10:31:00.050",
    }


def test1() -> None:
    from market.native.dolphin import NativeDolphinSubscription, OfficialDolphinDbDriver
    sessions: list[FakeSession] = []

    def factory(**kwargs: Any) -> FakeSession:
        session = FakeSession(**kwargs)
        sessions.append(session)
        return session

    driver = OfficialDolphinDbDriver(session_factory=factory)
    driver.start(
        host="ddb.test",
        port=8848,
        username="demo",
        password="secret",
        streaming_port=0,
        keep_alive_seconds=30,
    )
    subscription = NativeDolphinSubscription(
        table_name="tickStream",
        action_name="exp5Native",
        columns=("Code", "LastPrice"),
        offset=-1,
        resub=True,
    )
    rows: list[dict[str, Any]] = []
    driver.subscribe(subscription, rows.append)

    session = sessions[0]
    assert session.init_kwargs == {"keepAliveTime": 30}
    assert session.calls[0] == ("connect", "ddb.test", 8848, "demo", "secret")
    assert session.calls[1] == ("enableStreaming",)
    subscribe_args = session.calls[2][1]
    assert subscribe_args["tableName"] == "tickStream"
    assert subscribe_args["actionName"] == "exp5Native"
    assert subscribe_args["offset"] == -1
    assert subscribe_args["resub"] is True

    session.callback(["rb2701", 3097.0])
    session.callback([["rb2701", 3098.0], ["rb2701", 3099.0]])
    assert rows == [
        {"Code": "rb2701", "LastPrice": 3097.0},
        {"Code": "rb2701", "LastPrice": 3098.0},
        {"Code": "rb2701", "LastPrice": 3099.0},
    ]

    driver.stop()
    assert ("unsubscribe", "ddb.test", 8848, "tickStream", "exp5Native") in session.calls
    assert session.calls[-1] == ("close",)
    print("DolphinDB native driver offline test OK")


def test2() -> None:
    from market.basic.base import DataType, InstrumentId, InstrumentMeta
    from market.stream.dolphin import DolphinDbMarketConverter
    instrument_id = InstrumentId.from_str("rb2701.SHFE")
    meta = InstrumentMeta(
        instrument_id=instrument_id,
        price_precision=0,
        size_precision=0,
        price_increment=Decimal("1"),
        multiplier=Decimal("10"),
        exchange="SHFE",
    )
    converter = DolphinDbMarketConverter()

    first = _tick(100, 3096, 0)
    second = _tick(104, 3097, 500)
    first_events = converter.convert_tick(first, instrument_id, meta)
    second_events = converter.convert_tick(second, instrument_id, meta)
    assert [item.data_type for item in first_events] == [DataType.QUOTE_TICK]
    assert [item.data_type for item in second_events] == [
        DataType.QUOTE_TICK,
        DataType.TRADE_TICK,
    ]
    trade = second_events[1].event
    assert trade.price.as_double() == 3097
    assert trade.size.as_double() == 4

    bar_events = converter.convert_bar(_bar(), instrument_id, meta, "1-MINUTE")
    assert [item.data_type for item in bar_events] == [DataType.BAR, DataType.CUSTOM_BAR]
    assert bar_events[0].event.close.as_double() == 3097
    assert bar_events[1].event.get_factor("open_interest") == 1_500_000
    assert converter.convert_bar(_bar(), instrument_id, meta, "1-MINUTE") == []
    print("DolphinDB converter offline test OK")

def test3() -> None:
    import threading

    from market.basic.base import DataType, InstrumentId, InstrumentMeta
    from market.stream.dolphin.config import DolphinDbConfig, DolphinDbStreamSpec
    from market.native.dolphin.driver import rows_from_message
    from market.stream.dolphin.feed import DolphinDbLiveDataFeed

    tick_stream = DolphinDbStreamSpec.tick("futuresTickStream", "exp5Tick")
    bar_stream = DolphinDbStreamSpec.bar("futuresBarStream", "exp5Bar")
    config = DolphinDbConfig(
        host="127.0.0.1",
        port=8848,
        username="admin",
        password="secret",
        streams=(tick_stream, bar_stream),
    )
    holder: dict[str, FakeDolphinDbDriver] = {}

    def factory() -> FakeDolphinDbDriver:
        driver = FakeDolphinDbDriver()
        holder["driver"] = driver
        return driver

    feed = DolphinDbLiveDataFeed(config, driver_factory=factory)
    instrument_id = InstrumentId.from_str("rb2701.SHFE")
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=0,
            size_precision=0,
            price_increment=Decimal("1"),
            multiplier=Decimal("10"),
            exchange="SHFE",
        ),
    )

    quotes: list[Any] = []
    trades: list[Any] = []
    bars: list[Any] = []
    custom_bars: list[Any] = []
    completed = threading.Event()

    def on_custom_bar(event: Any) -> None:
        custom_bars.append(event)
        if trades and bars:
            completed.set()

    feed.register_quote_tick_handler(quotes.append)
    feed.register_trade_tick_handler(trades.append)
    feed.register_bar_handler(bars.append)
    feed.register_custom_bar_handler(on_custom_bar)
    feed.subscribe(instrument_id, DataType.QUOTE_TICK)
    feed.subscribe(instrument_id, DataType.TRADE_TICK)
    feed.subscribe(instrument_id, DataType.BAR, bar_spec="1-MINUTE")
    feed.subscribe(instrument_id, DataType.CUSTOM_BAR, bar_spec="1-MINUTE")

    feed.connect()
    driver = holder["driver"]
    assert driver.started
    assert driver.start_args["host"] == "127.0.0.1"
    assert len(driver.handlers) == 2
    assert feed.wait_until_ready(0.1)
    driver.publish(tick_stream, _tick_row(volume=100, millisec=0, last=3096, bid_size=10))
    driver.publish(tick_stream, _tick_row(volume=104, millisec=500, last=3097, bid_size=12))
    driver.publish(bar_stream, _bar_row())
    assert completed.wait(1.0)

    assert len(quotes) == 2
    assert len(trades) == 1
    assert trades[0].price.as_double() == 3097
    assert trades[0].size.as_double() == 4
    assert len(str(trades[0].trade_id)) == 36
    assert len(bars) == 1
    assert bars[0].close.as_double() == 3097
    assert bars[0].volume.as_double() == 123
    assert len(custom_bars) == 1
    assert custom_bars[0].get_factor("open_interest") == 1_500_000
    assert custom_bars[0].get_factor("turnover") == 3_800_000

    # Replayed duplicate bars must not be emitted twice.
    driver.publish(bar_stream, _bar_row())
    threading.Event().wait(0.05)
    assert len(bars) == 1
    assert len(custom_bars) == 1

    feed.disconnect()
    assert driver.stopped

    single = list(range(len(tick_stream.columns)))
    assert len(rows_from_message(single, tick_stream.columns)) == 1
    assert len(rows_from_message([single, single], tick_stream.columns)) == 2
    print("DolphinDB live feed offline test OK")


def test4() -> None:
    """Stage 4: connect and enable streaming without subscribing to a table."""
    from market.native.dolphin import OfficialDolphinDbDriver

    host = _required_env("DDB_HOST")
    port = int(os.getenv("DDB_PORT", "8848"))
    streaming_port = int(os.getenv("DDB_STREAMING_PORT", "0"))
    driver = OfficialDolphinDbDriver()
    print(
        "启动 DolphinDB 连接探针: "
        f"host={host} port={port} streaming_port={streaming_port}",
    )
    try:
        driver.start(
            host=host,
            port=port,
            username=_required_env("DDB_USERNAME"),
            password=_required_env("DDB_PASSWORD"),
            streaming_port=streaming_port,
            keep_alive_seconds=int(os.getenv("DDB_KEEP_ALIVE_SECONDS", "60")),
        )
        print("DolphinDB connection and streaming lifecycle OK")
    finally:
        driver.stop()


def test5() -> None:
    """Stage 5: subscribe to the real tick stream and discover one cu row."""
    import threading

    from market.native.dolphin import NativeDolphinSubscription, OfficialDolphinDbDriver
    from market.stream.dolphin import TICK_COLUMNS

    host = _required_env("DDB_HOST")
    port = int(os.getenv("DDB_PORT", "8848"))
    table_name = _required_env("DDB_TICK_STREAM_TABLE")
    action_name = os.getenv("DDB_TICK_ACTION", "bomberTickRawProbe")
    timeout = float(os.getenv("DDB_RAW_TIMEOUT", "30"))
    exact_symbol = os.getenv("DDB_SYMBOL", "").strip()
    symbol_prefix = os.getenv("DDB_SYMBOL_PREFIX", "cu").strip()
    if not exact_symbol and not symbol_prefix:
        raise SystemExit("DDB_SYMBOL 和 DDB_SYMBOL_PREFIX 至少需要配置一个")

    received = threading.Event()
    first_row: dict[str, Any] = {}
    rows_seen = 0

    def on_row(row: Any) -> None:
        nonlocal rows_seen
        if received.is_set():
            return
        rows_seen += 1
        code = str(row.get("Code") or "").strip()
        if exact_symbol:
            matched = code.casefold() == exact_symbol.casefold()
        else:
            matched = code.casefold().startswith(symbol_prefix.casefold())
        if not matched:
            return
        first_row.update(row)
        received.set()

    subscription = NativeDolphinSubscription(
        table_name=table_name,
        action_name=action_name,
        columns=TICK_COLUMNS,
        offset=int(os.getenv("DDB_STREAM_OFFSET", "-1")),
        resub=_env_bool("DDB_STREAM_RESUB", True),
        batch_size=int(os.getenv("DDB_STREAM_BATCH_SIZE", "0")),
        throttle=float(os.getenv("DDB_STREAM_THROTTLE", "0.01")),
    )
    driver = OfficialDolphinDbDriver()
    print(
        "启动 DolphinDB Tick 原始流探针: "
        f"host={host} port={port} table={table_name} action={action_name} "
        f"symbol={exact_symbol or (symbol_prefix + '*')}",
    )
    try:
        driver.start(
            host=host,
            port=port,
            username=_required_env("DDB_USERNAME"),
            password=_required_env("DDB_PASSWORD"),
            streaming_port=int(os.getenv("DDB_STREAMING_PORT", "0")),
            keep_alive_seconds=int(os.getenv("DDB_KEEP_ALIVE_SECONDS", "60")),
        )
        driver.subscribe(subscription, on_row)
        if not received.wait(timeout):
            raise TimeoutError(
                f"{timeout:g} 秒内扫描了 {rows_seen} 条 {table_name} 新 Tick，"
                f"但没有匹配 {exact_symbol or (symbol_prefix + '*')}；"
                "请确认当前交易时段、合约代码以及流表列顺序",
            )
        print(
            "收到 DolphinDB 原始 Tick: "
            f"Code={first_row.get('Code')} "
            f"ActionDay={first_row.get('ActionDay')} "
            f"UpdateTime={first_row.get('UpdateTime')} "
            f"UpdateMillisec={first_row.get('UpdateMillisec')} "
            f"LastPrice={first_row.get('LastPrice')} "
            f"Volume={first_row.get('Volume')} "
            f"Bid={first_row.get('BidPrice1')}@{first_row.get('BidVolume1')} "
            f"Ask={first_row.get('AskPrice1')}@{first_row.get('AskVolume1')} "
            f"perPenetrationTime={first_row.get('perPenetrationTime')}",
        )
        if not exact_symbol:
            print(
                "已发现精确合约代码。下一步请写入 .env："
                f"DDB_SYMBOL={first_row.get('Code')}",
            )
        print("DolphinDB raw tick stream OK")
    finally:
        driver.stop()


def test6() -> None:
    """Stage 6: convert one exact cu stream into standard QuoteTick/TradeTick."""
    import threading

    from market.basic.base import DataType, InstrumentId, InstrumentMeta, QuoteTick, TradeTick
    from market.stream.dolphin import DolphinDbConfig, DolphinDbLiveDataFeed, DolphinDbStreamSpec

    symbol = _required_env("DDB_SYMBOL").strip()
    exchange = os.getenv("DDB_EXCHANGE", "SHFE").upper()
    instrument_id = InstrumentId.from_str(f"{symbol}.{exchange}")
    timeout = float(os.getenv("DDB_STANDARD_TIMEOUT", os.getenv("DDB_RAW_TIMEOUT", "30")))
    quote_received = threading.Event()
    trade_received = threading.Event()

    stream = DolphinDbStreamSpec.tick(
        table_name=_required_env("DDB_TICK_STREAM_TABLE"),
        action_name=os.getenv("DDB_STANDARD_ACTION", "bomberCuStandardProbe"),
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

    def on_quote(tick: QuoteTick) -> None:
        if quote_received.is_set():
            return
        if tick.bid_size.precision != tick.ask_size.precision:
            raise AssertionError("QuoteTick bid/ask size precision mismatch")
        print(
            "标准 QuoteTick: "
            f"{tick.instrument_id} bid={tick.bid_price}@{tick.bid_size} "
            f"ask={tick.ask_price}@{tick.ask_size} ts={tick.ts_event}",
        )
        quote_received.set()

    def on_trade(tick: TradeTick) -> None:
        if trade_received.is_set():
            return
        if tick.price.as_double() <= 0 or tick.size.as_double() <= 0:
            raise AssertionError(
                f"TradeTick price/size must be positive: {tick.price}@{tick.size}",
            )
        print(
            "标准 TradeTick: "
            f"{tick.instrument_id} price={tick.price} size={tick.size} "
            f"trade_id={tick.trade_id} ts={tick.ts_event}",
        )
        trade_received.set()

    feed.register_quote_tick_handler(on_quote)
    feed.register_trade_tick_handler(on_trade)
    feed.subscribe(instrument_id, DataType.QUOTE_TICK)
    feed.subscribe(instrument_id, DataType.TRADE_TICK)

    print(f"启动 DolphinDB 标准行情探针: {instrument_id}")
    try:
        feed.connect()
        if not feed.wait_until_ready(float(os.getenv("DDB_CONNECT_TIMEOUT", "15"))):
            raise TimeoutError("DolphinDB Feed 连接或订阅提交超时")
        if not quote_received.wait(timeout):
            raise TimeoutError(f"{timeout:g} 秒内未收到 {instrument_id} 标准 QuoteTick")
        if not trade_received.wait(timeout):
            raise TimeoutError(
                f"{timeout:g} 秒内未收到 {instrument_id} 标准 TradeTick；"
                "可能是期间累计 Volume 没有增加",
            )
        print("DolphinDB standard tick feed OK")
    finally:
        feed.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="逐阶段验证 DolphinDB 实时行情源")
    parser.add_argument(
        "--stage",
        type=int,
        choices=range(1, 7),
        default=5,
        help="验证阶段：1-3 离线，4 连接，5 cu 原始流，6 标准 Tick（默认 5）",
    )
    args = parser.parse_args()
    stages = {1: test1, 2: test2, 3: test3, 4: test4, 5: test5, 6: test6}
    stages[args.stage]()


if __name__ == "__main__":
    main()
