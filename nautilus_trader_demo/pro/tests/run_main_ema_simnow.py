#!/usr/bin/env python3
"""当前主力EMA的分阶段无网络验证；不读取凭据、不连接柜台、不报模拟盘订单。"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
from functools import lru_cache
from importlib import import_module
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGES = ("signal", "routing", "ctp", "regression")


@lru_cache(maxsize=1)
def components():
    """仅在子阶段加载项目依赖，集中入口可以完整记录失败阶段。"""
    import bomber.framework.trader as tr
    from bomber.framework.market.basic import base as market
    from bomber.framework.datahub.sector_roles import SectorRoleAssignment, SectorRoleStore
    strategy = import_module("demos.01_main_ema.strategy")
    online = import_module("demos.01_main_ema.live_runner")
    return SimpleNamespace(
        tr=tr, market=market, strategy=strategy, online=online,
        Assignment=SectorRoleAssignment, Store=SectorRoleStore,
        old=market.InstrumentId.from_str("rb2704.SHFE"),
        new=market.InstrumentId.from_str("rb2705.SHFE"),
    )


def assignment(*, symbol="rb2704", factor=1, effective=1, available=None, next_day=False):
    c = components()
    return c.Assignment(
        date(2026, 5, 19) if next_day else date(2026, 5, 18),
        date(2026, 5, 18) if next_day else date(2026, 5, 15),
        effective, effective if available is None else available,
        {"RB": {"main": symbol}}, {"RB": {"main": Decimal(str(factor))}},
    )


def bar(stamp, close=3100, instrument=None):
    c = components()
    instrument = c.old if instrument is None else instrument
    return c.market.make_bar(
        instrument_id=instrument, open=close, high=close + 1, low=close - 1,
        close=close, volume=10, ts_event=stamp,
        meta=c.market.InstrumentMeta(
            instrument_id=instrument, price_precision=0, size_precision=0,
            price_increment=Decimal(1), multiplier=Decimal(10),
            currency="CNY", exchange=str(instrument.venue),
        ), bar_type="1-MINUTE",
    )


class CapturingContext:
    def __init__(self):
        self.intents = []

    def submit(self, intent):
        self.intents.append(intent)


def signal_fixture(store=None, quantity=1):
    c = components()
    store = c.Store((assignment(),)) if store is None else store
    strategy = c.strategy.MainEmaStrategy(
        "main-ema-test", store,
        c.strategy.MainEmaConfig("RB", "SHFE", 2, 3, Decimal(quantity)),
    )
    context = CapturingContext()
    strategy._bind(context)
    return strategy, context


class SignalTests(unittest.TestCase):
    def test_warmup_direction_and_dedup(self):
        strategy, context = signal_fixture()
        strategy.on_bar("bars", bar(1))
        strategy.on_bar("bars", bar(2))
        self.assertEqual(context.intents, [])
        strategy.on_bar("bars", bar(3))
        self.assertEqual(context.intents[-1].targets["rb_main"], 1)
        strategy.on_bar("bars", bar(4))
        self.assertEqual(len(context.intents), 1)
        strategy.on_bar("bars", bar(5, 3000))
        self.assertEqual(context.intents[-1].targets["rb_main"], -1)
        self.assertEqual([x.revision for x in context.intents], [1, 2])

    def test_old_contract_wrong_venue_and_duplicate_do_not_advance(self):
        c = components()
        strategy, context = signal_fixture()
        strategy.on_bar("bars", bar(1, instrument=c.new))
        self.assertEqual(strategy.bars_used, 0)
        strategy.on_bar("bars", bar(1))
        strategy.on_bar("bars", bar(1, 3000))
        wrong = c.market.InstrumentId.from_str("rb2704.DCE")
        strategy.on_bar("bars", bar(2, instrument=wrong))
        strategy.on_bar("bars", bar(0))
        self.assertEqual(strategy.bars_used, 1)
        self.assertEqual(strategy.last_processed_ns, 1)
        strategy.on_bar("bars", bar(2))
        strategy.on_bar("bars", bar(3))
        self.assertEqual(len(context.intents), 1)

    def test_missing_snapshot_and_publication_gate(self):
        c = components()
        store = c.Store((assignment(effective=2, available=4),))
        strategy, context = signal_fixture(store)
        for stamp in (1, 2, 3):
            strategy.on_bar("bars", bar(stamp))
        self.assertEqual(strategy.bars_used, 0)
        self.assertEqual(strategy.unavailable_events, 3)
        for stamp in (4, 5, 6):
            strategy.on_bar("bars", bar(stamp))
        self.assertEqual(strategy.bars_used, 3)
        self.assertEqual(len(context.intents), 1)

    def test_missing_factor_is_not_one(self):
        c = components()
        row = c.Assignment(date(2026, 5, 18), date(2026, 5, 15), 1, 1,
                           {"RB": {"main": "rb2704"}}, {})
        strategy, context = signal_fixture(c.Store((row,)))
        strategy.on_bar("bars", bar(1))
        self.assertEqual(strategy.bars_used, 0)
        self.assertEqual(strategy.unavailable_events, 1)
        self.assertEqual(context.intents, [])

    def test_factor_only_changes_research_price(self):
        c = components()
        strategy, context = signal_fixture(c.Store((assignment(factor=2),)))
        event = bar(1)
        strategy.on_bar("bars", event)
        self.assertEqual(event.close.as_decimal(), 3100)
        self.assertEqual(strategy.fast.value, 6200)
        for stamp in (2, 3):
            strategy.on_bar("bars", bar(stamp))
        self.assertEqual(context.intents[-1].metadata["adjusted_close"], "6200")

    def test_roll_preserves_indicators_and_direction(self):
        c = components()
        store = c.Store((assignment(), assignment(symbol="rb2705", factor="0.5",
                                                   effective=4, next_day=True)))
        strategy, context = signal_fixture(store)
        for stamp in (1, 2, 3):
            strategy.on_bar("bars", bar(stamp))
        strategy.on_bar("bars", bar(4, 6200, c.new))
        self.assertEqual(strategy.bars_used, 4)
        self.assertEqual(strategy.last_main, "rb2705")
        self.assertEqual(strategy.fast.value, 3100)
        self.assertEqual(strategy.slow.value, 3100)
        self.assertEqual(len(context.intents), 1)


def feed_fixture():
    c = components()

    class ManualBarFeed(c.market.MarketDataFeed):
        def connect(self):
            self._is_connected = True

        def disconnect(self):
            self._is_connected = False

        def _on_subscription_added(self, request):
            pass

        def _on_subscription_removed(self, request):
            pass

    return ManualBarFeed(source_id="main-ema-memory-bars")


def assemble(store, client, positions, *, online=True, dynamic=False):
    c = components()
    strategy = c.strategy.MainEmaStrategy(
        "main-ema-test", store,
        c.strategy.MainEmaConfig("RB", "SHFE", 2, 3, Decimal(1)),
    )
    runner_type = c.online.MainEmaSimnowRunner if online and not dynamic else c.tr.UnifiedStrategyRunner
    mode = c.tr.RuntimeMode.HISTORICAL if dynamic else c.tr.RuntimeMode.LIVE
    runner = runner_type(mode, position_manager=positions)
    assignments = tuple(
        c.tr.ContractAssignment("rb_main", c.market.InstrumentId.from_str(
            f"{row.instrument('RB', 'main')}.SHFE"), row.effective_ns,
            row.available_ns, index)
        for index, row in enumerate(store.assignments, 1)
    )
    resolver = c.tr.ScheduledContractResolver(assignments)
    instruments = tuple(dict.fromkeys(x.instrument_id for x in assignments))
    runner.add_data_feed("bars", feed_fixture())
    runner.add_execution_client(client)
    if dynamic:
        # 跨合约链路仅在 HISTORICAL 验证，不绕过 LIVE 恢复与持久化限制。
        runner.add_strategy(strategy, data_bindings=tuple(
            c.tr.DataBinding(str(item), "bars", item, c.market.DataType.BAR, "1-MINUTE")
            for item in instruments), execution_routes=(
                c.tr.DynamicExecutionRoute("rb_main", client.client_id, resolver),))
    elif online:
        runner.add_main_strategy(strategy, feed_id="bars", instrument_id=instruments[0],
                                 client_id=client.client_id)
    else:
        runner.add_strategy(strategy, data_bindings=(
            c.tr.DataBinding(str(instruments[0]), "bars", instruments[0],
                             c.market.DataType.BAR, "1-MINUTE"),), execution_routes=(
                c.tr.ExecutionRoute("rb_main", client.client_id, instruments[0]),))
    return runner, strategy


def recording_fixture(*, online=True, store=None):
    c = components()
    positions = c.tr.PositionManager()
    positions.apply_account_snapshot("recording", {}, revision=1, ts_event=0)
    client = c.tr.RecordingExecutionClient("recording")
    store = c.Store((assignment(),)) if store is None else store
    runner, strategy = assemble(store, client, positions, online=online)
    runner.start()
    return runner, strategy, client, positions


class RoutingTests(unittest.TestCase):
    def test_default_runner_does_not_retry(self):
        runner, strategy, client, positions = recording_fixture(online=False)
        self.addCleanup(runner.stop)
        for stamp in (1, 2, 3, 4):
            runner.publish("bars", bar(stamp))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(runner.target_store.get(strategy.strategy_id).revision, 1)

    def test_online_retry_preserves_strategy_revision(self):
        c = components()
        runner, strategy, client, positions = recording_fixture()
        self.addCleanup(runner.stop)
        for stamp in (1, 2, 3):
            runner.publish("bars", bar(stamp))
        runner.publish("bars", bar(4))
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(runner.target_store.get(strategy.strategy_id).revision, 1)
        self.assertEqual(client.requests[-1].logical_targets["rb_main"], 1)
        self.assertEqual(client.requests[-1].revision, client.requests[0].revision)
        self.assertEqual(client.requests[-1].ts_event, 4)
        self.assertEqual(client.requests[-1].metadata["signal_ts_event"], 3)
        self.assertTrue(client.requests[-1].metadata["retained_target_retry"])
        self.assertEqual(runner.portfolio_coordinator.state().strategy_revisions[strategy.strategy_id], 1)
        runner.publish("bars", bar(4))
        self.assertEqual(len(client.requests), 2)
        positions.set_account_position(client.client_id, c.old, 1)
        runner.publish("bars", bar(5))
        self.assertEqual(len(client.requests), 2)
        runner.publish("bars", bar(6, 3000))
        self.assertEqual(client.requests[-1].revision, 2)
        self.assertEqual(client.requests[-1].logical_targets["rb_main"], -1)

    def test_working_wrong_contract_and_recovery_block_retry(self):
        c = components()
        runner, strategy, client, positions = recording_fixture()
        self.addCleanup(runner.stop)
        for stamp in (1, 2, 3):
            runner.publish("bars", bar(stamp))
        positions.set_working_quantity(client.client_id, c.old, 1)
        runner.publish("bars", bar(4))
        positions.set_working_quantity(client.client_id, c.old, 0)
        runner.publish("bars", bar(5, instrument=c.new))
        positions.mark_recovery_required(client.client_id)
        runner.publish("bars", bar(6))
        self.assertEqual(len(client.requests), 1)

    def test_expired_target_is_not_retried(self):
        runner, strategy, client, positions = recording_fixture()
        self.addCleanup(runner.stop)
        c = components()
        runner.submit(c.tr.TargetPortfolio(strategy.strategy_id, 1, 2,
                                           {"rb_main": 1}, deadline_ns=3))
        runner.publish("bars", bar(4))
        runner.continue_execution_target(strategy.strategy_id, "rb_main", 4,
                                         trigger_instrument_id=c.old)
        self.assertEqual(len(client.requests), 1)

    def test_retry_requires_market_instrument(self):
        runner, strategy, client, positions = recording_fixture()
        self.addCleanup(runner.stop)
        with self.assertRaisesRegex(ValueError, "真实合约行情"):
            runner.continue_execution_target(strategy.strategy_id, "rb_main", 1)

    def test_unpublished_timestamp_cannot_retry(self):
        c = components()
        runner, strategy, client, positions = recording_fixture()
        self.addCleanup(runner.stop)
        for stamp in (1, 2, 3):
            runner.publish("bars", bar(stamp))
        self.assertFalse(runner.continue_execution_target(
            strategy.strategy_id, "rb_main", 4, trigger_instrument_id=c.old))
        self.assertEqual(len(client.requests), 1)

    def test_live_dynamic_route_remains_forbidden(self):
        c = components()
        strategy, _ = signal_fixture()
        runner = c.tr.UnifiedStrategyRunner(c.tr.RuntimeMode.LIVE)
        resolver = c.tr.ScheduledContractResolver((
            c.tr.ContractAssignment("rb_main", c.old, 1, 1, 1),))
        with self.assertRaisesRegex(ValueError, "仅允许HISTORICAL"):
            runner.add_strategy(strategy, data_bindings=(), execution_routes=(
                c.tr.DynamicExecutionRoute("rb_main", "recording", resolver),))

    def test_main_change_closes_fixed_route_gate(self):
        c = components()
        store = c.Store((assignment(), assignment(symbol="rb2705", factor="0.5",
                                                   effective=4, next_day=True)))
        runner, strategy, client, positions = recording_fixture(store=store)
        self.addCleanup(runner.stop)
        for stamp in (1, 2, 3):
            runner.publish("bars", bar(stamp))
        with self.assertRaisesRegex(RuntimeError, "固定路由已闭闸"):
            runner.publish("bars", bar(4, 6200, c.new))
        self.assertTrue(positions.is_recovery_required(client.client_id))
        with self.assertRaisesRegex(RuntimeError, "固定路由已闭闸"):
            runner.publish("bars", bar(5))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(strategy.bars_used, 3)

    def test_missing_reference_blocks_retained_target(self):
        c = components()
        missing = c.Assignment(date(2026, 5, 19), date(2026, 5, 18), 4, 4,
                              {"RB": {"main": "rb2704"}}, {})
        runner, strategy, client, positions = recording_fixture(
            store=c.Store((assignment(), missing)))
        self.addCleanup(runner.stop)
        for stamp in (1, 2, 3, 4):
            runner.publish("bars", bar(stamp))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(strategy.unavailable_events, 1)

    def test_market_health_blocks_retry_until_recovery_confirmed(self):
        from bomber.framework.market.stream.health import StreamHealthMonitor
        runner, strategy, client, positions = recording_fixture()
        self.addCleanup(runner.stop)
        for stamp in (1, 2, 3):
            runner.publish("bars", bar(stamp))
        monitor = StreamHealthMonitor("main-ema-memory-bars")
        monitor.on_connected()
        monitor.on_event(bar(3))
        runner.market_health_gate.register_feed("bars", monitor.snapshot)
        monitor.on_stream_interrupted("test interruption")
        runner.market_health_gate.on_feed_health("bars", monitor.snapshot)
        runner.publish("bars", bar(4))
        self.assertEqual(len(client.requests), 1)
        monitor.on_event(bar(5))
        runner.market_health_gate.on_feed_health("bars", monitor.snapshot)
        runner.publish("bars", bar(5))
        self.assertTrue(runner.market_health_snapshot(strategy.strategy_id).confirmation_required)
        self.assertEqual(len(client.requests), 1)


def ctp_fixture(*, quantity=1, rollover=False):
    """复用原生Driver、旧限价规划器和受控客户端，仅传输层是内存替身。"""
    c = components()
    from tests.run_p4_ctp_driver import FakeTraderTransport
    from bomber.framework.trader.execution.ctp import CtpLimitPlanner
    from bomber.framework.trader.execution.ctp.native_driver import CtpNativeTraderDriver

    class EmptyTransport(FakeTraderTransport):
        def query_positions(self):
            return {}

    transport = EmptyTransport()
    holder = {}
    driver = CtpNativeTraderDriver(
        "ctp-demo", "demo-account", transport, enable_test_orders=True,
        disconnect_handler=lambda reason: holder["client"].mark_disconnected(reason),
    )
    positions = c.tr.PositionManager()
    prices = c.tr.MarketReferencePriceStore()
    ledger = c.tr.CtpPositionLedger()
    backend = c.tr.NautilusLiveExecutionBackend("ctp-demo", driver)
    client = c.tr.ControlledLiveExecutionClient(
        "ctp-demo", CtpLimitPlanner(ledger, positions, prices, Decimal(1), 1),
        backend, positions,
        c.tr.PreTradeRiskManager("ctp-demo", positions, prices,
            default_limits=c.tr.RiskLimits(
                max_order_quantity=quantity, max_abs_position=quantity,
                max_order_notional=1000000, max_abs_position_notional=1000000,
                contract_multiplier=10, max_market_age_ns=120000000000)),
        account_id="demo-account", demo_environment_check=lambda: True,
    )
    holder["client"] = client
    accounting = c.tr.CtpExecutionAccounting(ledger, {c.old: 10, c.new: 10})
    backend.register_report_handler(accounting.on_report)
    rows = (assignment(),)
    if rollover:
        rows += (assignment(symbol="rb2705", factor="0.5", effective=6, next_day=True),)
    runner, strategy = assemble(c.Store(rows), client, positions, dynamic=rollover)
    # 手数仅为测试参数，策略算法保持不变。
    strategy.config = c.strategy.MainEmaConfig("RB", "SHFE", 2, 3, Decimal(quantity))
    runner.add_market_observer(prices)
    runner.start()
    client.arm_demo(client.DEMO_CONFIRMATION)
    return SimpleNamespace(runner=runner, strategy=strategy, client=client,
        positions=positions, ledger=ledger, driver=driver, transport=transport)


def raw_order(fields, **extra):
    return {
        "BrokerID": "9999", "InvestorID": "demo",
        "InstrumentID": fields["InstrumentID"], "ExchangeID": fields["ExchangeID"],
        "OrderRef": fields["OrderRef"], "FrontID": 1, "SessionID": 2,
        "OrderSysID": "SYS" + fields["OrderRef"], **extra,
    }


def fill(fixture, index, *, quantity=None, trade_id=None, accepted=True):
    fields = fixture.transport.sent[index]
    if accepted:
        fixture.transport.on_order(raw_order(fields, OrderStatus="3"))
    fixture.transport.on_trade(raw_order(
        fields, TradeID=trade_id or f"T{index}",
        Volume=fields["VolumeTotalOriginal"] if quantity is None else quantity,
        Price=fields["LimitPrice"],
    ))


def warmup(fixture):
    for stamp in (1, 2, 3):
        fixture.runner.publish("bars", bar(stamp))
    fill(fixture, 0)


class CtpTests(unittest.TestCase):
    def fixture(self, **kwargs):
        value = ctp_fixture(**kwargs)
        self.addCleanup(value.runner.stop)
        return value

    def test_current_strategy_open_close_reverse_without_signal_repeat(self):
        c = components()
        f = self.fixture()
        warmup(f)
        f.runner.publish("bars", bar(4, 3000))
        self.assertEqual(len(f.transport.sent), 2)
        self.assertEqual(f.transport.sent[1]["CombOffsetFlag"], "3")
        fill(f, 1)
        self.assertEqual(f.positions.account_position("ctp-demo", c.old), 0)
        f.runner.publish("bars", bar(5, 3000))
        self.assertEqual(len(f.transport.sent), 3)
        self.assertEqual(f.transport.sent[2]["Direction"], "1")
        self.assertEqual(f.transport.sent[2]["CombOffsetFlag"], "0")
        fill(f, 2)
        f.runner.publish("bars", bar(6, 3000))
        self.assertEqual(len(f.transport.sent), 3)
        self.assertEqual(f.runner.target_store.get(f.strategy.strategy_id).revision, 2)
        self.assertEqual(f.positions.account_position("ctp-demo", c.old), -1)
        self.assertEqual(f.ledger.snapshot(c.old).net_position, -1)
        self.assertEqual(f.positions.working_quantity("ctp-demo", c.old), 0)
        self.assertEqual(f.client.report_errors, ())

    def test_latest_signal_wins_before_pending_open(self):
        f = self.fixture()
        warmup(f)
        f.runner.publish("bars", bar(4, 3000))
        fill(f, 1)
        f.runner.publish("bars", bar(5, 4000))
        self.assertEqual(len(f.transport.sent), 3)
        self.assertEqual(f.transport.sent[2]["Direction"], "0")
        self.assertEqual(f.transport.sent[2]["CombOffsetFlag"], "0")
        self.assertEqual(f.runner.target_store.get(f.strategy.strategy_id).revision, 3)
        fill(f, 2)

    def test_partial_fill_and_duplicate_do_not_open_early(self):
        c = components()
        f = self.fixture(quantity=2)
        warmup(f)
        f.runner.publish("bars", bar(4, 3000))
        fill(f, 1, quantity=1, trade_id="part1")
        fill(f, 1, quantity=1, trade_id="part1", accepted=False)
        self.assertEqual(f.positions.account_position("ctp-demo", c.old), 1)
        self.assertEqual(f.ledger.snapshot(c.old).net_position, 1)
        f.runner.publish("bars", bar(5, 3000))
        self.assertEqual(len(f.transport.sent), 2)
        fill(f, 1, quantity=1, trade_id="part2", accepted=False)
        f.runner.publish("bars", bar(6, 3000))
        self.assertEqual(len(f.transport.sent), 3)
        self.assertEqual(f.transport.sent[2]["VolumeTotalOriginal"], 2)
        fill(f, 2)
        self.assertEqual(f.positions.account_position("ctp-demo", c.old), -2)
        self.assertEqual(f.client.report_errors, ())

    def test_canceled_remaining_close_is_replanned(self):
        c = components()
        f = self.fixture(quantity=2)
        warmup(f)
        f.runner.publish("bars", bar(4, 3000))
        fill(f, 1, quantity=1, trade_id="part1")
        f.transport.on_order(raw_order(f.transport.sent[1], OrderStatus="5"))
        self.assertEqual(f.positions.working_quantity("ctp-demo", c.old), 0)
        f.runner.publish("bars", bar(5, 3000))
        self.assertEqual(len(f.transport.sent), 3)
        self.assertEqual(f.transport.sent[2]["CombOffsetFlag"], "3")
        self.assertEqual(f.transport.sent[2]["VolumeTotalOriginal"], 1)
        fill(f, 2)
        f.runner.publish("bars", bar(6, 3000))
        self.assertEqual(f.transport.sent[3]["CombOffsetFlag"], "0")
        self.assertEqual(f.transport.sent[3]["VolumeTotalOriginal"], 2)
        fill(f, 3)
        self.assertEqual(f.client.report_errors, ())

    def test_roll_closes_old_then_opens_new(self):
        c = components()
        f = self.fixture(rollover=True)
        warmup(f)
        # 有旧仓时，新合约行情不得推动旧合约撤单/平仓。
        f.runner.publish("bars", bar(6, 6200, c.new))
        self.assertEqual(len(f.transport.sent), 1)
        self.assertEqual(f.runner.roll_coordinator.state(
            f.strategy.strategy_id, "rb_main").phase, c.tr.RollPhase.ACTIVE)
        # 第一根旧合约行情进入撤单阶段；下一根确认无在途后才平旧仓。
        f.runner.publish("bars", bar(7))
        self.assertEqual(len(f.transport.sent), 1)
        self.assertEqual(f.runner.roll_coordinator.state(
            f.strategy.strategy_id, "rb_main").phase, c.tr.RollPhase.CANCELING)
        f.runner.publish("bars", bar(8))
        self.assertEqual(len(f.transport.sent), 2)
        self.assertEqual(f.runner.roll_coordinator.state(
            f.strategy.strategy_id, "rb_main").phase, c.tr.RollPhase.CLOSING)
        self.assertEqual(f.transport.sent[1]["InstrumentID"], "rb2704")
        self.assertEqual(f.transport.sent[1]["CombOffsetFlag"], "3")
        # 旧平仓单仍在途时，即便新合约报价更新，也不能提前开仓。
        f.runner.publish("bars", bar(9, 6200, c.new))
        self.assertEqual(len(f.transport.sent), 2)
        self.assertEqual(f.positions.account_position("ctp-demo", c.new), 0)
        fill(f, 1)
        self.assertEqual(f.positions.account_position("ctp-demo", c.old), 0)
        self.assertEqual(f.positions.working_quantity("ctp-demo", c.old), 0)
        f.runner.publish("bars", bar(10, 6200, c.new))
        self.assertEqual(len(f.transport.sent), 3)
        self.assertEqual(f.transport.sent[2]["InstrumentID"], "rb2705")
        self.assertEqual(f.transport.sent[2]["CombOffsetFlag"], "0")
        fill(f, 2)
        self.assertEqual(f.positions.account_position("ctp-demo", c.old), 0)
        self.assertEqual(f.positions.account_position("ctp-demo", c.new), 1)
        self.assertEqual(f.strategy.fast.value, 3100)
        self.assertEqual(f.runner.target_store.get(f.strategy.strategy_id).revision, 1)
        self.assertEqual(f.client.report_errors, ())

    def test_disconnect_blocks_remaining_target(self):
        f = self.fixture()
        warmup(f)
        f.runner.publish("bars", bar(4, 3000))
        fill(f, 1)
        f.client.mark_disconnected("test_disconnect")
        f.runner.publish("bars", bar(5, 3000))
        self.assertEqual(len(f.transport.sent), 2)
        self.assertFalse(f.client.is_armed)

    def test_live_main_change_disarms_ctp(self):
        c = components()
        f = self.fixture()
        warmup(f)
        f.strategy.data_hub = c.Store((assignment(), assignment(
            symbol="rb2705", factor="0.5", effective=4, next_day=True)))
        with self.assertRaisesRegex(RuntimeError, "固定路由已闭闸"):
            f.runner.publish("bars", bar(4, 6200, c.new))
        self.assertFalse(f.client.is_armed)
        self.assertTrue(f.positions.is_recovery_required(f.client.client_id))
        self.assertEqual(len(f.transport.sent), 1)
        self.assertEqual(f.positions.account_position(f.client.client_id, c.old), 1)


REGRESSIONS = (
    ("tests/run_dynamic_routes.py",),
    ("tests/run_market_health_gate.py", "--stage", "all"),
    ("tests/run_execution_dispatch.py", "--stage", "all"),
    ("tests/run_ctp_ema_staging.py",),
    ("tests/run_p4_ctp_driver.py",),
    ("tests/run_p3_controlled_live.py", "--stage", "all"),
    ("tests/run_single_ema_online.py", "--stage", "offline"),
)


def run_process(command, timeout):
    started = time.monotonic()
    try:
        result = subprocess.run([sys.executable, *command], cwd=ROOT, timeout=timeout)
        code = result.returncode
    except subprocess.TimeoutExpired:
        print(f"超时：{timeout} 秒；该阶段未通过", flush=True)
        code = 1
    print(f"退出码={code} 耗时={time.monotonic() - started:.3f}s", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--timeout", type=int, default=180, help="每个子阶段／回归脚本的秒数上限")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("超时必须大于零")
    print(f"主力EMA增量验证 stage={args.stage} Python={sys.executable} 项目={ROOT}", flush=True)
    if args.stage == "all":
        for stage in STAGES:
            print(f"\n开始阶段：{stage}", flush=True)
            if run_process([str(Path(__file__).resolve()), "--stage", stage,
                            "--timeout", str(args.timeout)],
                           args.timeout * len(REGRESSIONS) + 30 if stage == "regression" else args.timeout):
                print(f"停止：{stage} 未通过。先修复本阶段，再继续后续模块。", flush=True)
                return 1
        print("全部无网络增量与相关回归通过；仍需当前SimNow行情及柜台验收。", flush=True)
        return 0
    if args.stage == "regression":
        for command in REGRESSIONS:
            print(f"\n回归：{' '.join(command)}", flush=True)
            if run_process(list(command), args.timeout):
                return 1
        return 0
    c = components()
    import bomber
    print(f"Bomber={getattr(bomber, '__version__', '未知')} 路径={bomber.__file__}", flush=True)
    print(f"策略={c.strategy.__file__} Runner={import_module('bomber.framework.trader.runner').__file__}", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        {"signal": SignalTests, "routing": RoutingTests, "ctp": CtpTests}[args.stage])
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
