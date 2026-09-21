"""第二类阶段1/2：多Bar同步和完整组合目标；不读文件、不联网。"""

from __future__ import annotations

import argparse
from decimal import Decimal

from examples.cross_section.strategy import CrossSectionConfig, CrossSectionMomentumStrategy
from market.basic.base import (
    Bar,
    DataType,
    InstrumentId,
    InstrumentMeta,
    MarketDataFeed,
    SubscriptionRequest,
    make_bar,
)
from strategy import DataBinding, ExecutionRoute, RecordingExecutionClient, RuntimeMode, UnifiedStrategyRunner
from strategy.bar_sync import BarSynchronizer


INSTRUMENTS = tuple(InstrumentId.from_str(name) for name in (
    "rb2610.SHFE", "RM609.CZCE", "SA609.CZCE", "CF609.CZCE", "m2609.DCE",
))
MULTIPLIERS = (10, 10, 20, 5, 10)
METAS = {
    item: InstrumentMeta(item, price_precision=0, size_precision=0,
                         price_increment=Decimal(5 if item.symbol.value.upper().startswith("CF") else 1),
                         multiplier=Decimal(multiplier),
                         exchange=str(item.venue))
    for item, multiplier in zip(INSTRUMENTS, MULTIPLIERS)
}


def _bar(instrument_id: InstrumentId, minute: int, close: int) -> Bar:
    step = int(METAS[instrument_id].price_increment)
    return make_bar(
        instrument_id, close, close + step, close - step, close, 1,
        1_800_000_000_000_000_000 + minute * 60_000_000_000,
        meta=METAS[instrument_id], bar_type="1-MINUTE",
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


def test1_synchronizer() -> None:
    """第五路到齐才输出；重复不重发，缺路不前填充，旧Bar明确拒绝。"""
    sync = BarSynchronizer(INSTRUMENTS)
    for item in INSTRUMENTS[:-1]:
        assert sync.push(_bar(item, 0, 100)) is None
    frame = sync.push(_bar(INSTRUMENTS[-1], 0, 100))
    assert frame is not None and len(frame.bars) == 5
    assert sync.push(_bar(INSTRUMENTS[0], 0, 100)) is None
    for item in INSTRUMENTS[:-1]:
        assert sync.push(_bar(item, 1, 101)) is None
    # 一路缺失，下一分钟到达时不能拿上一分钟的旧Bar凑帧。
    assert sync.push(_bar(INSTRUMENTS[0], 2, 102)) is None
    for item in INSTRUMENTS[1:]:
        completed = sync.push(_bar(item, 2, 102))
    assert completed is not None and completed.ts_event == _bar(INSTRUMENTS[0], 2, 102).ts_event
    try:
        sync.push(_bar(INSTRUMENTS[0], 1, 101))
    except ValueError as error:
        assert "时间回退" in str(error)
    else:
        raise AssertionError("回退Bar不得污染同步帧")
    print("第二类阶段1通过：五路同时间戳同步、缺路、重复和回退边界正常")


def test2_strategy_targets() -> None:
    """25个完整帧后，按20期收益率选强弱，一次提交五腿完整目标。"""
    feed = ManualBarFeed("CROSS_SECTION_TEST")
    client = RecordingExecutionClient("recording-only")
    strategy = CrossSectionMomentumStrategy(
        "cross-section-test",
        CrossSectionConfig(
            INSTRUMENTS,
            {item: Decimal(multiplier) for item, multiplier in zip(INSTRUMENTS, MULTIPLIERS)},
        ),
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL)
    runner.add_data_feed("five-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "five-bars", item, DataType.BAR, "1-MINUTE")
            for item in INSTRUMENTS
        ),
        execution_routes=tuple(
            ExecutionRoute(str(item), client.client_id, item) for item in INSTRUMENTS
        ),
    )
    runner.start()
    try:
        for minute in range(25):
            closes = (100 + 2 * minute, 100 + minute, 100, 200, 100 - 2 * minute)
            for item, close in zip(INSTRUMENTS, closes):
                feed.push(_bar(item, minute, close))
        assert strategy.synchronized_frames == 25
        assert strategy.rebalances == 1
        assert len(client.requests) == 1
        request = client.requests[0]
        assert len(request.targets) == 5
        assert request.targets[INSTRUMENTS[0]] > 0
        assert request.targets[INSTRUMENTS[-1]] < 0
        assert all(request.targets[item] == 0 for item in INSTRUMENTS[1:-1])
        assert request.metadata["long_instrument"] == str(INSTRUMENTS[0])
        assert request.metadata["short_instrument"] == str(INSTRUMENTS[-1])
        feed.push(_bar(INSTRUMENTS[0], 24, 148))
        assert len(client.requests) == 1
        # 下一批完整帧让强弱对调：原多腿变空、原空腿变多，其余仍显式归零。
        for minute in range(25, 30):
            closes = (
                max(10, 148 - 28 * (minute - 24)),
                100 + minute,
                100,
                200,
                52 + 28 * (minute - 24),
            )
            for item, close in zip(INSTRUMENTS, closes):
                feed.push(_bar(item, minute, close))
        assert strategy.synchronized_frames == 30
        assert strategy.rebalances == 2
        assert len(client.requests) == 2
        reversal = client.requests[-1]
        assert reversal.targets[INSTRUMENTS[0]] < 0
        assert reversal.targets[INSTRUMENTS[-1]] > 0
        assert all(reversal.targets[item] == 0 for item in INSTRUMENTS[1:-1])
    finally:
        runner.stop()
    print("第二类阶段2通过：同一策略完成排名、名义金额换手数和五腿原子目标快照")


def test3_fractional_contracts() -> None:
    """BN式小数合约必须按数量步长截断，不能被CTP的一手下限放大。"""
    instruments = INSTRUMENTS[:2]
    feed = ManualBarFeed("FRACTIONAL_TEST")
    client = RecordingExecutionClient("recording-fractional")
    strategy = CrossSectionMomentumStrategy(
        "fractional-test",
        CrossSectionConfig(
            instruments,
            {item: Decimal(1) for item in instruments},
            size_increments={item: Decimal("0.00000001") for item in instruments},
            lookback=1,
            rebalance_interval=1,
            target_notional=Decimal(10),
        ),
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL)
    runner.add_data_feed("fractional-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "fractional-bars", item, DataType.BAR, "1-MINUTE")
            for item in instruments
        ),
        execution_routes=tuple(
            ExecutionRoute(str(item), client.client_id, item) for item in instruments
        ),
    )
    runner.start()
    try:
        for minute, closes in enumerate(((100, 100), (110, 90))):
            for item, close in zip(instruments, closes):
                feed.push(_bar(item, minute, close))
        assert len(client.requests) == 1
        targets = client.requests[0].targets
        assert targets[instruments[0]] == Decimal("0.09090909")
        assert targets[instruments[1]] == Decimal("-0.11111111")
    finally:
        runner.stop()
    print("第二类阶段3通过：BN式小数目标严格按数量步长截断")


STAGES = {1: test1_synchronizer, 2: test2_strategy_targets, 3: test3_fractional_contracts}


def main() -> None:
    parser = argparse.ArgumentParser(description="第二类截面策略离线分阶段测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for test in selected.values():
        test()


if __name__ == "__main__":
    main()
