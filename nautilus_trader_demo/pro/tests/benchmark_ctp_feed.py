"""Offline throughput benchmark for the production CTP market-data path.

This benchmark deliberately excludes network and CTP front latency.  It measures:

1. ``CtpTickConverter``: CTP dict -> QuoteTick/TradeTick.
2. ``CtpLiveDataFeed``: conversion -> queue -> dispatcher -> handlers.

Run this on the same server and Python environment used for live trading.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import threading
import time
from decimal import Decimal
from typing import Any

from market.basic.base import DataType, InstrumentId, InstrumentMeta
from market.stream.ctp import CtpLiveDataFeed, CtpMdConfig, CtpTickConverter


class _NoopCtpDriver:
    """Driver used to run the production Feed without a network connection."""

    def __init__(self, callbacks: Any) -> None:
        self.callbacks = callbacks

    def start(self, **_: Any) -> None:
        return None

    def stop(self) -> None:
        return None

    def login(self, **_: Any) -> None:
        return None

    def subscribe(self, _: str) -> None:
        return None

    def unsubscribe(self, _: str) -> None:
        return None


def _meta() -> InstrumentMeta:
    instrument_id = InstrumentId.from_str("rb2701.SHFE")
    return InstrumentMeta(
        instrument_id=instrument_id,
        price_precision=0,
        size_precision=0,
        price_increment=Decimal("1"),
        multiplier=Decimal("10"),
        currency="CNY",
        exchange="SHFE",
    )


def _snapshot() -> dict[str, Any]:
    return {
        "TradingDay": "20260918",
        "ActionDay": "20260918",
        "InstrumentID": "rb2701",
        "ExchangeID": "SHFE",
        "UpdateTime": "10:30:00",
        "UpdateMillisec": 500,
        "LastPrice": 3096.0,
        "Volume": 1_000_000,
        "BidPrice1": 3095.0,
        "BidVolume1": 1000,
        "AskPrice1": 3096.0,
        "AskVolume1": 800,
        "OpenInterest": 1_500_000.0,
    }


def _advance(row: dict[str, Any], index: int) -> None:
    # Every frame changes both the quote and cumulative volume. After the first
    # baseline frame this creates one QuoteTick and one TradeTick per snapshot.
    row["Volume"] = 1_000_000 + index
    row["BidVolume1"] = 1000 + (index & 255)
    row["AskVolume1"] = 800 + ((index * 3) & 255)
    row["LastPrice"] = 3095.0 + (index & 1)


def _measure_converter(iterations: int) -> tuple[float, int]:
    meta = _meta()
    converter = CtpTickConverter()
    row = _snapshot()
    event_count = 0

    start = time.perf_counter_ns()
    for index in range(iterations):
        _advance(row, index)
        event_count += len(converter.convert(row, meta.instrument_id, meta))
    elapsed = (time.perf_counter_ns() - start) / 1_000_000_000
    return elapsed, event_count


def _measure_feed(iterations: int, timeout: float) -> tuple[float, int]:
    meta = _meta()
    expected_events = iterations * 2 - 1
    received = 0
    completed = threading.Event()

    def on_event(_: Any) -> None:
        nonlocal received
        # StreamDataFeed has a single dispatcher, so this counter has one writer.
        received += 1
        if received == expected_events:
            completed.set()

    feed = CtpLiveDataFeed(
        config=CtpMdConfig(
            front="tcp://127.0.0.1:1",
            broker_id="benchmark",
            user_id="benchmark",
            password="benchmark",
            flow_path="/tmp/bomber-ctp-benchmark",
        ),
        source_id="CTP_BENCHMARK",
        queue_size=expected_events + 16,
        driver_factory=_NoopCtpDriver,
    )
    feed.register_instrument(meta)
    feed.register_quote_tick_handler(on_event)
    feed.register_trade_tick_handler(on_event)
    feed.subscribe(meta.instrument_id, DataType.QUOTE_TICK)
    feed.subscribe(meta.instrument_id, DataType.TRADE_TICK)

    row = _snapshot()
    feed.connect()
    try:
        start = time.perf_counter_ns()
        for index in range(iterations):
            _advance(row, index)
            feed.on_depth_market_data(row)
        if not completed.wait(timeout):
            raise TimeoutError(
                f"dispatcher timeout: received={received}, expected={expected_events}",
            )
        elapsed = (time.perf_counter_ns() - start) / 1_000_000_000
    finally:
        feed.disconnect()
    return elapsed, received


def _run_case(name: str, function: Any, iterations: int, repeats: int) -> None:
    elapsed_values: list[float] = []
    event_counts: list[int] = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            elapsed, event_count = function(iterations)
        finally:
            gc.enable()
        elapsed_values.append(elapsed)
        event_counts.append(event_count)

    rates = [iterations / elapsed for elapsed in elapsed_values]
    event_rates = [count / elapsed for count, elapsed in zip(event_counts, elapsed_values)]
    print(f"\n{name}")
    print(f"  iterations:       {iterations:,} x {repeats}")
    print(f"  snapshots/sec:    median={statistics.median(rates):,.0f} min={min(rates):,.0f}")
    print(
        "  output events/sec: "
        f"median={statistics.median(event_rates):,.0f} min={min(event_rates):,.0f}",
    )
    print(
        "  us/snapshot:      "
        f"median={1_000_000 / statistics.median(rates):.2f} "
        f"worst={1_000_000 / min(rates):.2f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.iterations < 2 or args.repeats < 1 or args.warmup < 1:
        parser.error("iterations >= 2, repeats >= 1 and warmup >= 1 are required")

    print("CTP offline market-data benchmark")
    print("Warm-up (not measured)...")
    _measure_converter(args.warmup)
    _measure_feed(args.warmup, args.timeout)

    _run_case("1) converter only", _measure_converter, args.iterations, args.repeats)
    _run_case(
        "2) full feed pipeline",
        lambda count: _measure_feed(count, args.timeout),
        args.iterations,
        args.repeats,
    )


if __name__ == "__main__":
    main()
