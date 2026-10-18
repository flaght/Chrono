"""Combined offline replay smoke test for CTP and Binance data."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from market.basic.base import DataType, InstrumentId, InstrumentMeta
from market.replay.base import DataLoadError, FileReplayFeed
from market.replay.parsers.bar import (
    BarColumns,
    BinanceKlineParser,
    BinanceMarketType,
    MappedBarParser,
    binance_instrument_id,
)
from market.replay.parsers.tick import CtpTickParser


CTP_TICK_PATH = Path(
    "/workspace/data/fut_tick/7050707549_-/2026/202607/20260701/"
    "rb2610_20260701.csv",
)
CTP_BAR_1M_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728/"
    "rb2608_20260728.feather",
)
BN_FUTURES_BAR_1M_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/binance_data/"
    "futures/um/klines/1m/BTCUSDT/2023-09-02.csv",
)
BN_FUTURES_BAR_1H_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/binance_data/"
    "futures/um/klines/1h/BTCUSDT/2024-03-04.csv",
)
BN_SPOT_BAR_1H_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/binance_data/"
    "spot/klines/1h/BTCUSDT/2024-03-04.csv",
)


def main() -> None:
    feed = FileReplayFeed(source_id="CTP_BINANCE_FILE_REPLAY")

    ctp_tick_id = InstrumentId.from_str("rb2610.SHFE")
    ctp_bar_id = InstrumentId.from_str("rb2608.XSGE")
    bn_futures_id = binance_instrument_id("BTCUSDT", BinanceMarketType.FUTURES)
    bn_spot_id = binance_instrument_id("BTCUSDT", BinanceMarketType.SPOT)

    _register_instruments(
        feed,
        ctp_tick_id=ctp_tick_id,
        ctp_bar_id=ctp_bar_id,
        bn_futures_id=bn_futures_id,
        bn_spot_id=bn_spot_id,
    )
    _add_sources(feed)
    _subscribe(
        feed,
        ctp_tick_id=ctp_tick_id,
        ctp_bar_id=ctp_bar_id,
        bn_futures_id=bn_futures_id,
        bn_spot_id=bn_spot_id,
    )

    try:
        feed.connect()
        events = feed.load_events()
        print(f"loaded events: {len(events):,}")
        if events:
            print(f"first event: {events[0]}")
            print(f"last event:  {events[-1]}")
        print(f"replay summary: {feed.replay()}")
    finally:
        feed.disconnect()


def _register_instruments(
    feed: FileReplayFeed,
    *,
    ctp_tick_id: InstrumentId,
    ctp_bar_id: InstrumentId,
    bn_futures_id: InstrumentId,
    bn_spot_id: InstrumentId,
) -> None:
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=ctp_tick_id,
            price_precision=0,
            size_precision=0,
            price_increment=Decimal("1"),
            exchange="SHFE",
            currency="CNY",
        ),
    )
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=ctp_bar_id,
            price_precision=0,
            size_precision=0,
            price_increment=Decimal("1"),
            exchange="XSGE",
            currency="CNY",
        ),
    )
    for instrument_id in (bn_futures_id, bn_spot_id):
        feed.register_instrument(
            InstrumentMeta(
                instrument_id=instrument_id,
                price_precision=2,
                size_precision=6,
                price_increment=Decimal("0.01"),
                exchange="BINANCE",
                currency="USDT",
            ),
        )


def _add_sources(feed: FileReplayFeed) -> None:
    feed.add_tick_csv(
        CTP_TICK_PATH,
        CtpTickParser(exchange="SHFE"),
    )
    feed.add_bar_feather(
        CTP_BAR_1M_PATH,
        MappedBarParser(
            columns=BarColumns(
                symbol="symbol",
                exchange="exchange",
                timestamp="datetime",
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                value="value",
                open_interest="open_interest",
                vwap="vwap",
            ),
            bar_spec="1-MINUTE",
            timezone="Asia/Shanghai",
        ),
    )

    _add_binance_bar_source(
        feed,
        BN_FUTURES_BAR_1M_PATH,
        BinanceKlineParser(
            symbol="BTCUSDT",
            market_type=BinanceMarketType.FUTURES,
            interval="1m",
        ),
    )
    _add_binance_bar_source(
        feed,
        BN_FUTURES_BAR_1H_PATH,
        BinanceKlineParser(
            symbol="BTCUSDT",
            market_type=BinanceMarketType.FUTURES,
            interval="1h",
        ),
    )
    _add_binance_bar_source(
        feed,
        BN_SPOT_BAR_1H_PATH,
        BinanceKlineParser(
            symbol="BTCUSDT",
            market_type=BinanceMarketType.SPOT,
            interval="1h",
        ),
    )


def _add_binance_bar_source(
    feed: FileReplayFeed,
    path: Path,
    parser: BinanceKlineParser,
) -> None:
    paths = _csv_paths(path)
    for csv_path in paths:
        feed.add_bar_csv(csv_path, parser)
        print(
            f"configured Binance {parser.market_type.value}: {csv_path} -> "
            f"{parser.instrument_id} ({parser.bar_spec})",
        )


def _csv_paths(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise DataLoadError(f"Binance source is not CSV: {path}")
        return (path,)
    if path.is_dir():
        paths = tuple(sorted(path.glob("*.csv")))
        if not paths:
            raise DataLoadError(f"Binance directory contains no CSV files: {path}")
        return paths
    raise DataLoadError(f"Binance source does not exist: {path}")


def _subscribe(
    feed: FileReplayFeed,
    *,
    ctp_tick_id: InstrumentId,
    ctp_bar_id: InstrumentId,
    bn_futures_id: InstrumentId,
    bn_spot_id: InstrumentId,
) -> None:
    feed.subscribe(ctp_tick_id, DataType.QUOTE_TICK)
    feed.subscribe(ctp_tick_id, DataType.TRADE_TICK)

    feed.subscribe(ctp_bar_id, DataType.BAR, bar_spec="1-MINUTE")
    feed.subscribe(
        ctp_bar_id,
        DataType.CUSTOM_BAR,
        bar_spec="1-MINUTE",
        fields=("value", "open_interest", "vwap"),
    )

    for instrument_id, bar_spec in (
        (bn_futures_id, "1-MINUTE"),
        (bn_futures_id, "1-HOUR"),
        (bn_spot_id, "1-HOUR"),
    ):
        feed.subscribe(instrument_id, DataType.BAR, bar_spec=bar_spec)
        feed.subscribe(
            instrument_id,
            DataType.CUSTOM_BAR,
            bar_spec=bar_spec,
            fields=(
                "quote_volume",
                "trade_count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
            ),
        )


if __name__ == "__main__":
    main()
