"""G1-G2：实时行情健康状态与异常检测的分阶段测试。

本文件不连接任何真实交易所。G1验证统一三态模型；G2验证无行情超时、
底层流中断、时间戳回退和异步消费队列溢出。
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

from market.basic.base import InstrumentId, SubscriptionRequest, make_quote_tick
from market.stream.base import StreamDataFeed
from market.stream.health import (
    MarketHealthReason,
    MarketHealthState,
    StreamHealthConfig,
    StreamHealthMonitor,
)


@dataclass(frozen=True)
class _Event:
    instrument_id: str
    ts_event: int


class _ProbeFeed(StreamDataFeed):
    """只启动公共分发线程、不启动网络连接的测试Feed。"""

    def __init__(self, queue_size: int = 1) -> None:
        super().__init__(
            "G_MARKET_HEALTH_PROBE",
            queue_size=queue_size,
            health_config=StreamHealthConfig(
                startup_grace_seconds=1,
                stale_after_seconds=1,
            ),
        )

    def _start_network_client(self) -> None:
        return None

    def _stop_network_client(self) -> None:
        return None

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        del request

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        del request


def test_g1() -> None:
    """G1：连接、首条行情和断开映射为统一三态。"""
    ticks = {"mono": 0, "wall": 1_000_000_000}
    monitor = StreamHealthMonitor(
        "G1",
        StreamHealthConfig(startup_grace_seconds=5, stale_after_seconds=5),
        monotonic_ns=lambda: ticks["mono"],
        wall_time_ns=lambda: ticks["wall"],
    )
    transitions = []
    monitor.register_handler(transitions.append)

    assert monitor.snapshot.state is MarketHealthState.DISCONNECTED
    monitor.on_connected()
    assert monitor.snapshot.state is MarketHealthState.DEGRADED
    assert monitor.snapshot.reason is MarketHealthReason.WAITING_FOR_FIRST_EVENT

    assert monitor.on_event(_Event("rb2701.SHFE", 100))
    assert monitor.snapshot.state is MarketHealthState.READY
    assert monitor.snapshot.reason is MarketHealthReason.HEALTHY

    monitor.on_disconnected()
    assert monitor.snapshot.state is MarketHealthState.DISCONNECTED
    assert [item.state for item in transitions] == [
        MarketHealthState.DEGRADED,
        MarketHealthState.READY,
        MarketHealthState.DISCONNECTED,
    ]
    print("G1通过：实时Feed统一输出READY/DEGRADED/DISCONNECTED健康状态")


def test_g2_detection() -> None:
    """G2a：检测无数据超时、流中断和同一数据通道时间戳回退。"""
    ticks = {"mono": 0, "wall": 1_000_000_000}
    monitor = StreamHealthMonitor(
        "G2_DETECTION",
        StreamHealthConfig(startup_grace_seconds=2, stale_after_seconds=3),
        monotonic_ns=lambda: ticks["mono"],
        wall_time_ns=lambda: ticks["wall"],
    )
    monitor.on_connected()
    ticks["mono"] = 2_000_000_000
    assert monitor.check_timeout().reason is MarketHealthReason.STALE

    ticks["mono"] = 2_100_000_000
    assert monitor.on_event(_Event("rb2701.SHFE", 200))
    assert monitor.snapshot.state is MarketHealthState.READY

    monitor.on_stream_interrupted("测试网络断流")
    assert monitor.snapshot.reason is MarketHealthReason.STREAM_INTERRUPTED
    assert monitor.snapshot.stream_interruptions == 1
    assert monitor.on_event(_Event("rb2701.SHFE", 201))
    assert monitor.snapshot.state is MarketHealthState.READY

    assert not monitor.on_event(_Event("rb2701.SHFE", 199))
    assert monitor.snapshot.reason is MarketHealthReason.TIMESTAMP_ROLLBACK
    assert monitor.snapshot.timestamp_rollbacks == 1
    assert monitor.acknowledge_degradation().state is MarketHealthState.READY
    print("G2a通过：无数据超时、流中断和时间戳回退检测正常")


def test_g2_queue_overflow() -> None:
    """G2b：公共StreamDataFeed队列溢出会降级且保留计数。"""
    feed = _ProbeFeed(queue_size=1)
    instrument_id = InstrumentId.from_str("BTCUSDT.BINANCE")
    handler_entered = threading.Event()
    release_handler = threading.Event()

    def blocking_handler(event: object) -> None:
        del event
        handler_entered.set()
        release_handler.wait(1.0)

    feed.register_quote_tick_handler(blocking_handler)
    feed.connect()
    try:
        def quote(ts_event: int):
            return make_quote_tick(
                instrument_id=instrument_id,
                bid_price=100,
                ask_price=101,
                bid_size=1,
                ask_size=1,
                ts_event=ts_event,
                ts_init=ts_event,
            )

        feed.enqueue_event(quote(100))
        assert handler_entered.wait(1.0)
        feed.enqueue_event(quote(101))
        feed.enqueue_event(quote(102))
        snapshot = feed.health_snapshot
        assert snapshot.state is MarketHealthState.DEGRADED
        assert snapshot.reason is MarketHealthReason.QUEUE_OVERFLOW
        assert snapshot.queue_overflows == 1
    finally:
        release_handler.set()
        time.sleep(0.05)
        feed.disconnect()
    assert feed.health_state is MarketHealthState.DISCONNECTED
    print("G2b通过：队列溢出不再静默发生，Feed会降级并累计丢包数")


def main() -> None:
    parser = argparse.ArgumentParser(description="G1-G2行情健康状态分阶段验证")
    parser.add_argument("--stage", type=int, choices=range(1, 4))
    args = parser.parse_args()
    stages = {1: test_g1, 2: test_g2_detection, 3: test_g2_queue_overflow}
    if args.stage is not None:
        stages[args.stage]()
        return
    for stage in stages.values():
        stage()


if __name__ == "__main__":
    main()
