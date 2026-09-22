"""第六类逐步验证：目标计划、独立时钟、Runner 路由和历史事件顺序。"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from datahub.target_schedule import TargetPlan, TargetScheduleStore


def _schedule() -> TargetScheduleStore:
    return TargetScheduleStore((
        TargetPlan(100, 50, {"rb": Decimal(2), "sa": Decimal(-1)}, "v1"),
        TargetPlan(200, 150, {"rb": Decimal(0)}, "v2"),
    ))


def test1_plan() -> None:
    from datahub.target_schedule import TargetPlanUnavailable
    from examples.scheduled_targets.local_input import load_target_csv

    schedule = _schedule()
    assert schedule.slots_between(-1, 100) == (100,)
    assert schedule.at(100).targets["sa"] == -1
    try:
        schedule.at(200, as_of_ns=100)
    except TargetPlanUnavailable:
        pass
    else:
        raise AssertionError("未来发布的计划不能被提前读取")
    try:
        TargetScheduleStore((TargetPlan(100, 0, {"rb": 1}), TargetPlan(100, 0, {"rb": 2})))
    except ValueError:
        pass
    else:
        raise AssertionError("同一时点重复计划必须拒绝")
    sample = Path(__file__).resolve().parents[1] / "examples/scheduled_targets/positions_20260728.csv"
    parsed = load_target_csv(sample)
    assert len(parsed.slots) == 2 and parsed.plans[0].targets["sa"] == -1
    mixed = load_target_csv(sample.with_name("mixed_contract_fixture.csv"))
    assert len(mixed.slots) == 2
    assert mixed.plans[0].metadata["instrument_types"] == (
        ("CSI300", "INDEX"), ("IF2609", "FUTURE"), ("IO2609-C-4000", "OPTION"),
    )
    print("S1通过：完整目标、发布时间、重复时点与因果读取正常")


def test2_strategy() -> None:
    from examples.scheduled_targets.scheduled_strategy import ScheduledTargetStrategy
    class Context:
        def __init__(self) -> None:
            self.intents = []

        def submit(self, intent) -> None:
            self.intents.append(intent)

        def position(self, target_key: str) -> Decimal:
            return Decimal(0)

    context = Context()
    strategy = ScheduledTargetStrategy("schedule-test", _schedule())
    strategy._bind(context)
    strategy._start()
    strategy.on_time(100)
    strategy.on_time(100)
    strategy.on_time(200)
    assert len(context.intents) == 2
    assert dict(context.intents[0].targets) == {"rb": 2, "sa": -1}
    assert dict(context.intents[1].targets) == {"rb": 0}
    assert context.intents[1].update_mode.value == "REPLACE"
    assert [item.status for item in strategy.audit] == ["SUBMITTED", "SUBMITTED"]
    strategy._stop()
    late = ScheduledTargetStrategy("late", _schedule())
    late_context = Context()
    late._bind(late_context)
    late._start()
    late.on_time(150)
    assert not late_context.intents and late.audit[0].status == "MISSED"
    late._stop()
    print("S2通过：独立时钟精确触发、全量替换、防重和迟到不追单正常")


def test3_runner() -> None:
    from examples.scheduled_targets.scheduled_strategy import ScheduledTargetStrategy
    from market.basic.base import InstrumentId
    from strategy.contracts import ExecutionRoute, RuntimeMode
    from strategy.execution.recording import RecordingExecutionClient
    from strategy.runner import UnifiedStrategyRunner
    from strategy.scheduling import ManualClockFeed

    feed = ManualClockFeed()
    client = RecordingExecutionClient("recording")
    strategy = ScheduledTargetStrategy("runner-schedule", _schedule())
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed("clock", feed)
    runner.add_execution_client(client)
    runner.add_strategy(strategy, data_bindings=(), time_feed_ids=("clock",), execution_routes=(
        ExecutionRoute("rb", "recording", InstrumentId.from_str("rb2609.SHFE")),
        ExecutionRoute("sa", "recording", InstrumentId.from_str("SA609.CZCE")),
    ))
    try:
        runner.start()
        feed.emit_time(100)
        feed.emit_time(200)
        assert len(client.requests) == 2
        assert client.requests[0].logical_targets["sa"] == -1
        assert client.requests[1].logical_targets["sa"] == 0
        assert client.requests[1].metadata["source_revision"] == "v2"
    finally:
        runner.stop()
    print("S3通过：无Bar也能定时提交；REPLACE经Portfolio和ExecutionRoute显式清旧腿")


def test4_replay_order() -> None:
    from market.basic.base import InstrumentId, make_bar
    from strategy.scheduling import TimedFileReplayFeed

    instrument = InstrumentId.from_str("rb2609.SHFE")
    events = (
        make_bar(instrument, 1, 1, 1, 1, 1, 100),
        make_bar(instrument, 2, 2, 2, 2, 1, 100),
        make_bar(instrument, 3, 3, 3, 3, 1, 300),
    )

    class InMemoryReplay(TimedFileReplayFeed):
        def connect(self) -> None:
            self._is_connected = True

        def load_events(self, force_reload: bool = False):
            return events

    feed = InMemoryReplay((100, 200, 400))
    order: list[str] = []
    feed.register_bar_handler(lambda bar: order.append(f"bar:{bar.ts_init}"))
    feed.register_time_handler(lambda slot: order.append(f"time:{slot}"))
    feed.connect()
    summary = feed.replay()
    assert summary.bars == 3
    assert order == ["bar:100", "bar:100", "time:100", "time:200", "bar:300"]
    print("S4通过：同时间戳行情先于计划；末尾无行情时不伪造触发")


def test6_execution_audit() -> None:
    from examples.scheduled_targets.execution_audit import audit_schedule_execution

    symbols = {"rb": "rb2609.SHFE", "sa": "SA609.CZCE"}

    def fill(slot: int, ts: int, key: str, side: str, qty: int):
        return SimpleNamespace(
            report_type="FILLED", client_order_id=f"{slot}-{key}",
            instrument_id=symbols[key], order_side=side,
            filled_quantity=Decimal(qty), fill_price=Decimal(100), ts_event=ts,
            metadata={"schedule_slot_ns": slot, "commission": "1.00 CNY"},
        )

    reports = (
        fill(100, 101, "rb", "BUY", 2),
        fill(100, 102, "sa", "SELL", 1),
        fill(200, 201, "rb", "SELL", 2),
        fill(200, 202, "sa", "BUY", 1),
    )
    flat = {symbol: Decimal(0) for symbol in symbols.values()}
    result = audit_schedule_execution(
        _schedule(), reports, symbols,
        actual_positions=flat, working_positions=flat, native_positions=flat,
    )
    assert len(result.fills) == 4
    assert result.commission_by_currency == {"CNY": Decimal(4)}
    for bad in (fill(100, 100, "rb", "BUY", 2),
                fill(100, 101, "rb", "BUY", 1)):
        try:
            audit_schedule_execution(
                _schedule(), (bad, *reports[1:]), symbols,
                actual_positions=flat, working_positions=flat, native_positions=flat,
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("同Bar或数量错误的成交必须被审计拒绝")
    print("S6通过：逐时点净变仓、严格下一时点、费用和双仓位终态审计正常")


STAGES = {"plan": test1_plan, "strategy": test2_strategy,
          "runner": test3_runner, "replay": test4_replay_order,
          "audit": test6_execution_audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    for name, function in STAGES.items():
        if args.stage in (name, "all"):
            function()


if __name__ == "__main__":
    main()
