"""04完整策略与实际CTP传输的无网络装配验证；全部MD/TD使用假API。"""

from contextlib import redirect_stderr
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from importlib import import_module
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bomber.framework.datahub.sector_roles import SectorRoleAssignment
from bomber.framework.market.basic.base import DataType, make_bar, make_trade_tick
from bomber.framework.market.stream.ctp.feed import CtpDepthObservation, CtpLiveDataFeed, CtpMdConfig
from bomber.framework.trader import (StrategyTemplate, ExecutionRoute, UnifiedStrategyRunner,
                                     RuntimeMode, RecordingExecutionClient, PositionManager)
from bomber.framework.trader.execution.ctp import CtpNativeTraderDriver, CtpTdApiTransport, CtpLimitPlanner
from bomber.framework.trader import RiskLimits
from bomber.framework.trader.execution.builders import build_ctp_portfolio_execution
from bomber.framework.trader.runtime.ctp import assert_flat_account

# 只导入夹具模块，不把其TestCase导入当前命名空间重复收集旧测试。
fixtures = import_module("tests.run_main_ema_live")
live = import_module("demos.04_cross_section.run_live")
runtime = import_module("demos.04_cross_section.live_runtime")
recovery = import_module("demos.04_cross_section.recovery")
BASE, MINUTE = fixtures.BASE, fixtures.MINUTE


class RejectFirstApi(fixtures.FlatTdApi):
    def reqOrderInsert(self, data, reqid):
        super().reqOrderInsert(data, reqid)
        self.onRspOrderInsert(data, {"ErrorID": 42, "ErrorMsg": "fake reject"}, reqid, True)
        return 0


class CrossSectionLiveTests(unittest.TestCase):
    def restored_session(self):
        original = self.session(orders=True)
        self.frame(original, 0, 3100, 3000)
        self.frame(original, 1, 3130, 2970)
        self.fill(original, 0)
        self.fill(original, 1)
        original.manager.save()
        original.runner.stop()
        transport = CtpTdApiTransport(client_id=runtime.CLIENT_ID, account_id="demo",
            front="tcp://fake:1234", broker_id="9999", investor_id="demo", password="fake",
            app_id="fake-app", auth_code="fake-auth", td_api_base=fixtures.FlatTdApi,
            flow_path=str(self.root / "resume-td"), timeout_seconds=0.05)
        holder = {}
        driver = CtpNativeTraderDriver(runtime.CLIENT_ID, "demo", transport,
            enable_simnow_orders=True, max_session_orders=20,
            disconnect_handler=lambda reason: holder["session"].runner.close_role_gate(reason))
        self.addCleanup(driver.stop)
        restored = runtime.assemble(original.args, original.references, driver, fixtures.ManualMd())
        holder["session"] = restored
        restored.manager.restore()
        return restored

    def test_resume_loads_ownership_revision_and_does_not_retry_before_warmup(self):
        s = self.restored_session()
        revision = recovery.validate_restored_session(s, "20260922")
        self.assertEqual(s.strategy._revision, revision)
        self.assertEqual(s.runner.position_manager.position(runtime.CLIENT_ID, "rb2704.SHFE"), 1)
        self.assertEqual(s.runner.position_manager.position(runtime.CLIENT_ID, "hc2704.SHFE"), -1)
        self.assertEqual(s.strategy.synchronized_frames, 0)
        s.runner.accept_bars = True
        with patch.object(s.runner, "continue_execution_target") as continuation:
            s.runner.continue_on_tick(make_trade_tick(s.references.instrument_ids["RB"], 3130, 1,
                ts_event=self.now, meta=s.references.instrument_meta("RB"), trade_id="resume-warm"))
            continuation.assert_not_called()
        submitted = []
        s.strategy._bind(SimpleNamespace(submit=submitted.append))
        s.strategy.set_targets({str(i): Decimal(0) for i in s.runner.fixed_ids}, self.now)
        self.assertEqual(submitted[0].revision, revision + 1)

    def test_failed_resume_shutdown_preserves_original_file(self):
        s = self.restored_session()
        before = s.manager.repository.path.read_bytes()
        controller = runtime.CrossSectionLifecycle(s.driver.transport, md_front="tcp://fake:1234",
            orders=True, resume=True, expected_trading_day="20260922")
        controller.driver = s.driver
        result = controller.shutdown(s, None)
        self.assertTrue(result["resume_state_preserved"])
        self.assertEqual(s.manager.repository.path.read_bytes(), before)

    def test_resume_validates_gross_and_today_yesterday_buckets(self):
        s = self.restored_session()
        gross, rows = {}, []
        for instrument in s.runner.fixed_ids:
            ledger = s.ledger.snapshot(instrument)
            gross[str(instrument)] = (ledger.long_total, ledger.short_total)
            rows.append({"InstrumentID": str(instrument.symbol), "ExchangeID": str(instrument.venue),
                "HedgeFlag": "1", "PositionDate": "1",
                "PosiDirection": "2" if ledger.net_position > 0 else "3", "Position": "1"})
        transport = SimpleNamespace(query_gross_positions=lambda: gross, query_position_details=lambda: rows)
        s.client = SimpleNamespace(client_id=runtime.CLIENT_ID,
            account_state=SimpleNamespace(balances={"CNY": SimpleNamespace(available=Decimal(1000000))}))
        with patch.object(s.driver, "reconcile_active_orders", return_value=SimpleNamespace(orders=())):
            recovery.validate_account(s, transport)
            rows[0]["PositionDate"] = "2"
            with self.assertRaisesRegex(RuntimeError, "今昨仓分桶"):
                recovery.validate_account(s, transport)

    def test_resume_rejects_changed_trading_day_and_contracts(self):
        s = self.restored_session()
        with self.assertRaisesRegex(RuntimeError, "相同交易日"):
            recovery.validate_restored_session(s, "20260923")
        s.runner.fixed_ids = frozenset((s.references.instrument_ids["RB"],))
        with self.assertRaisesRegex(RuntimeError, "主力组合"):
            recovery.validate_restored_session(s, "20260922")

    def test_resume_rejects_unfinished_local_orders(self):
        s = self.session(orders=True)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        with self.assertRaisesRegex(RuntimeError, "活动订单已结束"):
            recovery.validate_restored_session(s, "20260922")

    def test_resume_rejects_account_position_mismatch(self):
        s = self.restored_session()
        # 未进行新的权威账户对账，不得凭保存的持仓开闸。
        with patch.object(s.runner.position_manager, "account_position", return_value=Decimal(0)):
            with self.assertRaisesRegex(RuntimeError, "柜台净仓"):
                recovery.validate_account(s, SimpleNamespace())

    def test_resume_cli_requires_original_existing_state(self):
        filename = self.root / "original.json"
        flags = ("--mode", "simnow", "--enable-orders", "--confirm-simnow", "--state-file", str(filename))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            live.parse_args(self.command(*flags, "--resume"))
        filename.write_text("{}")
        self.assertTrue(live.parse_args(self.command(*flags, "--resume")).resume)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            live.parse_args(self.command(*flags))

    def test_live_execution_uses_current_time_with_newer_quotes_than_signal_bar(self):
        s = self.session(orders=True)
        self.frame(s, 0, 3100, 3000)
        stamp = BASE + 2 * MINUTE - 1
        self.now = stamp + 2_000_000_001
        s.references.refresh()
        for p, price in (("RB", 3130), ("HC", 2970)):
            tick = make_trade_tick(s.references.instrument_ids[p], price, 1,
                ts_event=stamp + 1_500_000_001, meta=s.references.instrument_meta(p), trade_id=p)
            s.runner.observe_tick(tick)
            UnifiedStrategyRunner.publish(s.runner, "ctp-execution-ticks", tick)
        self.bar(s, "RB", stamp, 3130)
        self.bar(s, "HC", stamp, 2970)
        self.assertEqual(len(s.transport._api.sent), 2)
        target = s.runner.target_store.get(runtime.CLIENT_ID)
        self.assertEqual(target.ts_event, self.now)
        self.assertEqual(target.metadata["signal_bar_ns"], stamp)

    def setUp(self):
        temp = TemporaryDirectory(prefix="cross-section-offline-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.now = BASE
        clock = patch("time.time_ns", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.assignment = SectorRoleAssignment(date(2026, 9, 22), date(2026, 9, 21), BASE, BASE,
            {"RB": {"main": "rb2704"}, "HC": {"main": "hc2704"}},
            {"RB": {"main": Decimal(2)}, "HC": {"main": Decimal(1)}})
        self.specs = {p: SimpleNamespace(symbol=f"{p.lower()}2704", product=p, venue="SHFE",
            currency="CNY", tick=Decimal(tick), multiplier=Decimal(multiplier))
            for p, tick, multiplier in (("RB", 1, 10), ("HC", 10, 10))}
        self.service = SimpleNamespace(products=("RB", "HC"), trading_day=date(2026, 9, 22),
            started_ns=BASE, factor_date_basis="source", refresh=lambda: None,
            snapshot=lambda ns: self.assignment,
            instrument_specs={p: {"main": spec} for p, spec in self.specs.items()},
            manifest={"factor_date": "2026-09-21", "ready": True})

    def command(self, *extra):
        return ["--connect", "--products", "RB,HC", "--expected-trading-day", "20260922",
            "--expected-source-day", "20260921", "--expected-instrument", "RB=rb2704.SHFE",
            "--expected-instrument", "HC=hc2704.SHFE", *extra]

    def session(self, orders=False, api_base=fixtures.FlatTdApi, **options):
        transport = CtpTdApiTransport(client_id=runtime.CLIENT_ID, account_id="demo",
            front="tcp://fake:1234", broker_id="9999", investor_id="demo", password="fake",
            app_id="fake-app", auth_code="fake-auth", td_api_base=api_base,
            flow_path=str(self.root / "td"), timeout_seconds=0.05)
        holder = {}
        driver = CtpNativeTraderDriver(runtime.CLIENT_ID, "demo", transport,
            enable_simnow_orders=orders, max_session_orders=20,
            disconnect_handler=lambda reason: holder["s"].runner.close_role_gate(reason))
        self.addCleanup(driver.stop)
        refs = runtime.PortfolioReferences(self.service)
        args = SimpleNamespace(mode="simnow" if orders else "recording",
            products=self.service.products, expected_instruments={p: str(i) for p, i in refs.instrument_ids.items()},
            lookback=1, rebalance_interval=2, group_fraction=Decimal("0.30"),
            target_notional=Decimal(35000), max_quantity=Decimal(2),
            max_notional=Decimal(50000), limit_offset_ticks=1, state_file=self.root / "state.json")
        for name, value in options.items():
            setattr(args, name, value)
        s = runtime.assemble(args, refs, driver, fixtures.ManualMd())
        holder["s"] = s
        s.transport, s.args = transport, args
        if not orders:
            driver.start(lambda report: None)
        s.runner.start()
        self.addCleanup(s.runner.stop)
        assert_flat_account(driver, transport)
        self.raw_ticks(s, BASE, {p: 3100 if p == "RB" else 3000 for p in self.service.products})
        if orders:
            s.client.arm_demo(s.client.DEMO_CONFIRMATION)
        s.runner.accept_bars = True
        return s

    def raw_ticks(self, s, stamp, prices):
        self.now = stamp
        s.references.refresh()  # 模拟控制线程刷新；行情回调自身不得访问数据库。
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        # 测试按整分钟跳时钟；先模拟两腿边界前持续Tick心跳，再依次触发聚合。
        # 生产行情没有此预填路径，实际逐腿Tick回调独立更新时间。
        if hasattr(s, "runner"):
            for p, value in prices.items():
                s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], value, 1,
                    ts_event=stamp, meta=s.references.instrument_meta(p), trade_id=f"heartbeat-{p}-{stamp}"))
        for p, value in prices.items():
            s.upstream._emit_trade_tick(make_trade_tick(s.references.instrument_ids[p], value, 1,
                ts_event=stamp, ts_init=stamp, meta=s.references.instrument_meta(p), trade_id=f"{p}-{stamp}"))

    def bar(self, s, p, stamp, price):
        s.references.refresh()  # 手动跳时钟的夹具模拟控制线程更新缓存。
        s.runner.publish("ctp-bars", make_bar(s.references.instrument_ids[p], price, price,
            price, price, 1, stamp, meta=s.references.instrument_meta(p)))

    def frame(self, s, minute, rb, hc):
        stamp = BASE + minute * MINUTE + MINUTE - 1
        # Fresh ticks只用于时效闸门，不经聚合，以便严格控制同步Bar到达次序。
        self.now = stamp + 1
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        for p in s.references.products:
            s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], rb if p == "RB" else hc,
                1, ts_event=self.now, meta=s.references.instrument_meta(p), trade_id=f"fresh-{p}-{minute}"))
        self.bar(s, "RB", stamp, rb)
        self.bar(s, "HC", stamp, hc)
        return stamp

    def fill(self, s, index, duplicate=False, volume=None, trade_suffix=""):
        fields = s.transport._api.sent[index]
        raw = {"BrokerID": "9999", "InvestorID": "demo", "InstrumentID": fields["InstrumentID"],
            "ExchangeID": fields["ExchangeID"], "OrderRef": fields["OrderRef"],
            "OrderSysID": f"SYS{index}", "FrontID": 1, "SessionID": 2}
        s.transport._api.onRtnOrder({**raw, "OrderStatus": "3"})
        trade = {**raw, "TradeID": f"T{index}{trade_suffix}",
            "Volume": volume or fields["VolumeTotalOriginal"], "Price": fields["LimitPrice"]}
        key = f"{fields['InstrumentID']}.{fields['ExchangeID']}"
        long, short = s.transport._api.gross.get(key, (0, 0))
        amount = trade["Volume"]
        if fields["CombOffsetFlag"] == "0":
            if fields["Direction"] == "0":
                long += amount
            else:
                short += amount
        elif fields["Direction"] == "0":
            short -= amount
        else:
            long -= amount
        s.transport._api.gross[key] = (long, short)
        s.transport._api.onRtnTrade(trade)
        if duplicate:
            s.transport._api.onRtnTrade(trade)

    def test_cli_default_recording_and_explicit_orders(self):
        args = live.parse_args(self.command())
        self.assertEqual(args.mode, "recording")
        self.assertEqual(args.products, ("RB", "HC"))
        self.assertEqual(args.factor_availability, "observed-on-read")
        args = live.parse_args(self.command("--mode", "simnow", "--enable-orders", "--confirm-simnow",
            "--state-file", str(self.root / "new.json")))
        self.assertEqual(args.max_quantity, 1)

    def test_optional_expected_instruments_select_current_main(self):
        command = ["--connect", "--products", "CU,PB,NI,AU,AG",
            "--expected-trading-day", "20261012", "--expected-source-day", "20261009"]
        for extra in ([], ["--mode", "simnow", "--enable-orders", "--confirm-simnow",
                          "--state-file", str(self.root / "automatic.json")]):
            with self.subTest(extra=extra):
                args = live.parse_args([*command, *extra])
                self.assertEqual(args.expected_instruments, {})
                self.assertEqual(args.products, ("CU", "PB", "NI", "AU", "AG"))
        s = self.session(expected_instruments={})
        self.assertEqual({p: str(i) for p, i in s.runner.instruments.items()},
                         {"RB": "rb2704.SHFE", "HC": "hc2704.SHFE"})
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.assertEqual(len(s.client.requests), 1)

    def test_explicit_expected_instrument_mismatch_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "本次主力不符"):
            self.session(expected_instruments={"RB": "rb2705.SHFE", "HC": "hc2704.SHFE"})

    def test_cli_rejects_unsafe_and_invalid_inputs(self):
        old = self.root / "old.json"
        old.write_text("keep")
        cases = (("--mode", "simnow"), ("--enable-orders",), ("--products", "RB,RB"),
            ("--products", "RB,HC,NI"), ("--seconds", "nan"), ("--max-quantity", "1.5"),
            ("--group-fraction", "0.6"), ("--factor-date-basis", "trading"),
            ("--expected-source-day", "20261032"), ("--reference-refresh-seconds", "nan"),
            ("--mode", "simnow", "--enable-orders", "--confirm-simnow", "--state-file", str(old)))
        for extra in cases:
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                live.parse_args(self.command(*extra))
        self.assertEqual(old.read_text(), "keep")

    def test_partial_frame_duplicate_and_real_budget_targets(self):
        s = self.session()
        stamp = self.frame(s, 0, 3100, 3000)
        self.assertEqual(s.strategy.synchronized_frames, 1)
        self.bar(s, "RB", stamp, 3100)
        self.assertEqual(s.strategy.synchronized_frames, 1)
        self.now += MINUTE
        next_stamp = stamp + MINUTE
        for p in s.references.products:
            s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], 3100, 1,
                ts_event=self.now, trade_id=f"{p}-next", meta=s.references.instrument_meta(p)))
        self.bar(s, "RB", next_stamp, 3130)
        self.assertFalse(s.client.requests)
        self.bar(s, "HC", next_stamp, 2970)
        targets = s.client.requests[-1].targets
        self.assertEqual(targets[s.references.instrument_ids["RB"]], 1)
        self.assertEqual(targets[s.references.instrument_ids["HC"]], -1)
        self.assertEqual(s.strategy.last_signal.scores["RB"], Decimal(3130) / Decimal(3100) - 1)

    def test_delayed_leg_joins_previous_minute_after_fast_leg_advanced(self):
        s = self.session()
        first = BASE + MINUTE - 1
        second = first + MINUTE
        def ready(stamp):
            self.now = stamp + 1
            s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
            for p in s.references.products:
                s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], 3100, 1,
                    ts_event=self.now, meta=s.references.instrument_meta(p), trade_id=f"{p}-{stamp}"))
        ready(first)
        self.bar(s, "RB", first, 3100)
        self.bar(s, "RB", first, 3100)
        ready(second)
        self.bar(s, "RB", second, 3130)
        self.assertEqual(s.strategy.synchronized_frames, 0)
        self.bar(s, "HC", first, 3000)
        self.assertEqual(s.strategy.synchronized_frames, 1)
        self.assertEqual(s.strategy.last_emitted_ns, first)
        self.bar(s, "HC", second, 2970)
        self.assertEqual(s.strategy.synchronized_frames, 2)
        self.assertEqual(len(s.client.requests), 1)
        self.assertEqual(s.client.requests[-1].ts_event, second)
        self.assertEqual(s.runner.bar_counts[s.references.instrument_ids["RB"]], 2)
        self.assertFalse(s.runner.pending_frames)

    def test_incomplete_frame_expires_without_fabricating_missing_leg(self):
        s = self.session()
        first = BASE + MINUTE - 1
        self.now = first + 1
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        for p in s.references.products:
            s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], 3100, 1,
                ts_event=self.now, meta=s.references.instrument_meta(p), trade_id=f"start-{p}"))
        self.bar(s, "RB", first, 3100)
        self.now = first + 121_000_000_000
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        for p in s.references.products:
            s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], 3100, 1,
                ts_event=self.now, meta=s.references.instrument_meta(p), trade_id=f"end-{p}"))
        self.bar(s, "RB", self.now - 1, 3130)
        self.assertEqual(s.runner.expired_frame_count, 1)
        self.assertNotIn(first, s.runner.pending_frames)
        self.assertEqual(s.strategy.synchronized_frames, 0)
        self.assertFalse(s.client.requests)

    def test_reverse_close_fills_continue_on_real_ticks_without_new_frame(self):
        s = self.session(orders=True, rebalance_interval=1)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.fill(s, 0)
        self.fill(s, 1)
        self.frame(s, 2, 2990, 3110)
        self.assertEqual(len(s.transport._api.sent), 4)
        self.fill(s, 2)
        self.fill(s, 3)
        revision = s.runner.target_store.get(runtime.CLIENT_ID).revision
        frames = s.strategy.synchronized_frames
        self.now += 1_000_000_000
        ticks = []
        for p, price in (("RB", 2990), ("HC", 3110)):
            tick = make_trade_tick(s.references.instrument_ids[p], price, 1, ts_event=self.now,
                meta=s.references.instrument_meta(p), trade_id=f"continue-{p}")
            ticks.append(tick)
            s.upstream._emit_trade_tick(tick)
        self.assertEqual(len(s.transport._api.sent), 6)
        self.assertEqual(s.strategy.synchronized_frames, frames)
        self.assertEqual(s.runner.target_store.get(runtime.CLIENT_ID).revision, revision)
        for tick in ticks:
            s.upstream._emit_trade_tick(tick)
        self.assertEqual(len(s.transport._api.sent), 6)
        self.fill(s, 4)
        self.fill(s, 5)
        self.assertEqual(s.strategy.position("rb2704.SHFE"), -1)
        self.assertEqual(s.strategy.position("hc2704.SHFE"), 1)
        self.assertFalse(s.client.report_errors)

    def test_stale_tick_cannot_continue_retained_targets(self):
        s = self.session(orders=True, rebalance_interval=1)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.fill(s, 0)
        self.fill(s, 1)
        self.frame(s, 2, 2990, 3110)
        self.fill(s, 2)
        self.fill(s, 3)
        self.now += 11_000_000_000
        for p in s.references.products:
            tick = make_trade_tick(s.references.instrument_ids[p], 3100, 1,
                ts_event=self.now - 11_000_000_000, meta=s.references.instrument_meta(p), trade_id=p)
            s.runner.continue_on_tick(tick)
        self.assertEqual(len(s.transport._api.sent), 4)

    def test_zero_lot_budget_is_not_forced_to_one(self):
        s = self.session(target_notional=Decimal(100))
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.assertTrue(s.client.requests)
        self.assertTrue(all(value == 0 for value in s.client.requests[-1].targets.values()))

    def test_real_transport_multileg_fills_attribution_and_checkpoint(self):
        s = self.session(orders=True)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        sent = s.transport._api.sent
        self.assertEqual({o["InstrumentID"]: o["LimitPrice"] for o in sent},
                         {"rb2704": 3131.0, "hc2704": 2960.0})
        self.fill(s, 0, duplicate=True)
        self.fill(s, 1, duplicate=True)
        self.assertEqual(s.strategy.fills_received, 2)
        for p, qty in (("RB", 1), ("HC", -1)):
            i = s.references.instrument_ids[p]
            self.assertEqual(s.strategy.position(str(i)), qty)
            self.assertEqual(s.runner.position_manager.account_position(runtime.CLIENT_ID, i), qty)
            self.assertEqual(s.runner.position_manager.unassigned_position(runtime.CLIENT_ID, i), 0)
        payload = json.loads(s.args.state_file.read_text())["payload"]
        self.assertEqual(len(payload["positions"]["attribution"]["owned"]), 2)
        self.assertFalse(s.client.report_errors)

    def test_contract_specific_multiplier_controls_integer_lots(self):
        self.service.instrument_specs["HC"]["main"] = SimpleNamespace(
            **{**vars(self.specs["HC"]), "multiplier": Decimal(5)})
        s = self.session(orders=True, max_quantity=Decimal(3))
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.assertEqual({row["InstrumentID"]: row["VolumeTotalOriginal"]
                          for row in s.transport._api.sent}, {"rb2704": 1, "hc2704": 2})
        self.fill(s, 0)
        self.fill(s, 1)
        self.assertEqual(s.strategy.position("hc2704.SHFE"), -2)
        self.assertEqual(s.ledger.snapshot(s.references.instrument_ids["HC"]).short_total, 2)
        self.assertFalse(s.client.report_errors)

    def test_portfolio_factory_rejects_missing_terms_and_multiplier_mismatch(self):
        s = self.session()
        rb, hc = s.references.instrument_ids.values()
        with self.assertRaisesRegex(ValueError, "同一非空集合"):
            build_ctp_portfolio_execution(s.driver, trading_day="20260922", price_increments={rb: 1},
                multipliers={rb: 10, hc: 10}, instrument_limits={rb: RiskLimits(contract_multiplier=10)},
                session_check=lambda: True)
        with self.assertRaisesRegex(ValueError, "乘数不一致"):
            build_ctp_portfolio_execution(s.driver, trading_day="20260922", price_increments={rb: 1},
                multipliers={rb: 5}, instrument_limits={rb: RiskLimits(contract_multiplier=10)},
                session_check=lambda: True)
        for ticks in ({}, {rb: 0}, {rb: Decimal("NaN")}):
            with self.subTest(ticks=ticks), self.assertRaises(ValueError):
                CtpLimitPlanner(s.ledger, s.runner.position_manager, s.execution.prices, ticks, 1)

    def test_reverse_closes_then_retains_revision_on_unscheduled_frame(self):
        s = self.session(orders=True)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.fill(s, 0)
        self.fill(s, 1)
        self.frame(s, 2, 3130, 2970)
        self.frame(s, 3, 3000, 3100)
        self.assertEqual(len(s.transport._api.sent), 4)
        self.assertTrue(all(o["CombOffsetFlag"] == "3" for o in s.transport._api.sent[2:]))
        self.fill(s, 2)
        self.fill(s, 3)
        revision = s.runner.target_store.get(runtime.CLIENT_ID).revision
        self.frame(s, 4, 2990, 3110)
        self.assertEqual(len(s.transport._api.sent), 6)
        self.assertTrue(all(o["CombOffsetFlag"] == "0" for o in s.transport._api.sent[4:]))
        self.assertEqual(s.runner.target_store.get(runtime.CLIENT_ID).revision, revision)
        self.fill(s, 4)
        self.fill(s, 5)
        self.assertEqual(s.strategy.position("rb2704.SHFE"), -1)
        self.assertEqual(s.strategy.position("hc2704.SHFE"), 1)
        self.assertFalse(s.client.report_errors)

    def test_stale_one_leg_closes_gate_even_other_leg_is_fresh(self):
        s = self.session(orders=True)
        self.frame(s, 0, 3100, 3000)
        self.now += 11_000_000_000
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        s.runner.observe_tick(make_trade_tick(s.references.instrument_ids["RB"], 3100, 1,
            ts_event=self.now, trade_id="rb-only", meta=s.references.instrument_meta("RB")))
        self.assertFalse(s.runner.session_ready())
        with self.assertRaisesRegex(RuntimeError, "陈旧"):
            self.bar(s, "RB", self.now - 1, 3100)
        self.assertFalse(s.runner.accept_bars)
        self.assertFalse(s.transport._api.sent)

    def test_fresh_depth_without_new_trades_keeps_session_ready(self):
        s = self.session()
        previous = dict(s.runner.tick_events)
        self.now += 11_000_000_000
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        s.upstream.depth_observations = {i: CtpDepthObservation("20260922", self.now,
            time.monotonic_ns()) for i in s.runner.fixed_ids}
        self.assertTrue(s.runner.session_ready())
        self.assertEqual(s.runner.tick_events, previous)
        self.assertEqual(s.strategy.synchronized_frames, 0)
        self.assertFalse(s.transport._api.sent)

    def test_instrument_age_cli_default_override_and_invalid_values(self):
        self.assertEqual(live.parse_args(self.command()).instrument_max_age_seconds, 10)
        self.assertEqual(live.parse_args(self.command("--instrument-max-age-seconds", "30"))
                         .instrument_max_age_seconds, 30)
        for value in ("0", "-1", "nan", "inf", "121"):
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                live.parse_args(self.command("--instrument-max-age-seconds", value))

    def test_explicit_depth_age_limit_still_checks_both_clocks(self):
        s = self.session(instrument_max_age_seconds=30)
        self.now += 16_000_000_000
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        ids = list(s.runner.fixed_ids)
        received = time.monotonic_ns()
        s.upstream.depth_observations = {i: CtpDepthObservation("20260922", self.now,
            received) for i in ids}
        s.upstream.depth_observations[ids[0]] = CtpDepthObservation("20260922",
            self.now - 16_000_000_000, received - 15_000_000_000)
        self.assertTrue(s.runner.session_ready())
        for event_age, receive_age in ((31, 1), (1, 31)):
            with self.subTest(event_age=event_age, receive_age=receive_age):
                s.upstream.depth_observations[ids[0]] = CtpDepthObservation("20260922",
                    self.now - event_age * 1_000_000_000,
                    time.monotonic_ns() - receive_age * 1_000_000_000)
                self.assertFalse(s.runner.session_ready())
                details = s.runner.stale_instruments()[str(ids[0])]
                self.assertEqual(details["max_age_seconds"], 30)

    def test_expired_bar_is_skipped_and_later_fresh_frames_continue(self):
        s = self.session(rebalance_interval=1)
        self.now = BASE + 4 * MINUTE + 1
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        for p in s.references.products:
            s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], 3100, 1,
                ts_event=self.now, meta=s.references.instrument_meta(p), trade_id=f"fresh-{p}"))
        self.bar(s, "RB", BASE + MINUTE - 1, 3100)
        self.assertEqual(s.runner.expired_bar_count, 1)
        self.assertEqual(s.strategy.synchronized_frames, 0)
        self.assertIsNone(s.runner.failure)
        self.assertTrue(s.runner.accept_bars)
        self.assertFalse(s.transport._api.sent)
        self.frame(s, 4, 3100, 3000)
        self.frame(s, 5, 3110, 2990)
        self.assertEqual(s.strategy.synchronized_frames, 2)
        self.assertEqual(s.strategy.rebalances, 1)

    def test_future_bar_closes_gate_with_timestamp_diagnostic(self):
        s = self.session(orders=True)
        with self.assertRaisesRegex(RuntimeError, "来自未来.*rb2704.SHFE.*bar_ns.*now_ns"):
            self.bar(s, "RB", self.now + 1, 3100)
        self.assertFalse(s.runner.accept_bars)
        self.assertFalse(s.transport._api.sent)

    def test_expired_bar_callback_does_not_query_reference_source(self):
        s = self.session()
        stamp = BASE + MINUTE - 1
        self.now = stamp + 121_000_000_000
        s.references.refresh()
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        s.upstream.depth_observations = {i: CtpDepthObservation("20260922", self.now,
            time.monotonic_ns()) for i in s.runner.fixed_ids}
        with patch.object(s.references.service, "snapshot", side_effect=AssertionError("回调不可读库")) as read:
            s.runner.publish("ctp-bars", make_bar(s.references.instrument_ids["RB"],
                3100, 3100, 3100, 3100, 1, stamp, meta=s.references.instrument_meta("RB")))
            read.assert_not_called()
        self.assertEqual(s.runner.expired_bar_count, 1)
        self.assertEqual(s.runner.expired_bar_samples[0]["age_seconds"], 121)
        self.assertEqual(s.strategy.synchronized_frames, 0)
        self.assertFalse(s.runner.last_bars)
        self.assertIsNone(s.runner.failure)
        self.assertTrue(s.runner.accept_bars)

    def test_blocked_reference_refresh_does_not_block_bar_callbacks(self):
        s = self.session()
        stamp = BASE + MINUTE - 1
        self.now = stamp + 1
        s.references.refresh()
        for p in s.references.products:
            s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], 3100, 1,
                ts_event=self.now, meta=s.references.instrument_meta(p), trade_id=p))
        entered, release = threading.Event(), threading.Event()
        errors = []
        assignment = self.assignment
        def blocked(_):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("测试刷新未释放")
            return assignment
        def refresh():
            try:
                s.references.refresh()
            except Exception as error:
                errors.append(error)
        with patch.object(s.references.service, "snapshot", side_effect=blocked):
            worker = threading.Thread(target=refresh)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                for p in s.references.products:
                    s.runner.publish("ctp-bars", make_bar(s.references.instrument_ids[p],
                        3100, 3100, 3100, 3100, 1, stamp, meta=s.references.instrument_meta(p)))
                self.assertEqual(s.strategy.synchronized_frames, 1)
                self.assertTrue(worker.is_alive())
            finally:
                release.set()
                worker.join(2)
        self.assertFalse(errors)
        self.assertFalse(worker.is_alive())

    def test_reference_cache_expiry_and_refresh_failure_block_access(self):
        refs = runtime.PortfolioReferences(self.service)
        self.now += 31_000_000_000
        with self.assertRaisesRegex(RuntimeError, "过期"):
            refs.snapshot(self.now)
        refs.refresh()
        self.assertEqual(refs.snapshot(self.now), self.assignment)
        with patch.object(self.service, "snapshot", side_effect=RuntimeError("fake query failed")):
            with self.assertRaisesRegex(RuntimeError, "fake query failed"):
                refs.refresh()
        with self.assertRaisesRegex(RuntimeError, "fake query failed"):
            refs.snapshot(self.now)

    def test_depth_guard_checks_each_leg_day_event_receive_and_regression(self):
        s = self.session()
        ids = list(s.runner.fixed_ids)
        for bad in (None,
                CtpDepthObservation("20260921", self.now, time.monotonic_ns()),
                CtpDepthObservation("20260922", self.now - 11_000_000_000, time.monotonic_ns()),
                CtpDepthObservation("20260922", self.now + 1, time.monotonic_ns()),
                CtpDepthObservation("20260922", self.now, time.monotonic_ns() - 11_000_000_000),
                CtpDepthObservation("20260922", self.now, time.monotonic_ns(), True)):
            with self.subTest(bad=bad):
                s.upstream.depth_observations = {i: CtpDepthObservation("20260922", self.now,
                    time.monotonic_ns()) for i in ids}
                if bad is None:
                    del s.upstream.depth_observations[ids[0]]
                else:
                    s.upstream.depth_observations[ids[0]] = bad
                self.assertFalse(s.runner.session_ready())
                self.assertEqual(set(s.runner.stale_instruments()), {str(ids[0])})

    def test_ctp_unchanged_depth_updates_observation_without_duplicate_events(self):
        refs = runtime.PortfolioReferences(self.service)
        meta = refs.instrument_meta("RB")
        feed = CtpLiveDataFeed(CtpMdConfig("tcp://fake:1", "9999", "fake", "fake"))
        feed.register_instrument(meta)
        feed.subscribe(meta.instrument_id, DataType.QUOTE_TICK)
        feed.subscribe(meta.instrument_id, DataType.TRADE_TICK)
        stamp = datetime.fromtimestamp(BASE / 1e9, timezone(timedelta(hours=8)))
        row = {"InstrumentID": str(meta.instrument_id.symbol), "TradingDay": "20260922",
            "ActionDay": stamp.strftime("%Y%m%d"), "UpdateTime": stamp.strftime("%H:%M:%S"),
            "UpdateMillisec": 0, "Volume": 10, "LastPrice": 3100,
            "BidPrice1": 3099, "AskPrice1": 3100, "BidVolume1": 1, "AskVolume1": 1}
        with patch.object(feed, "enqueue_event") as enqueue:
            feed.on_depth_market_data(row)
            self.assertEqual(enqueue.call_count, 1)
            later = stamp + timedelta(seconds=11)
            row["UpdateTime"] = later.strftime("%H:%M:%S")
            feed.on_depth_market_data(row)
            self.assertEqual(enqueue.call_count, 1)
            observation = feed.depth_observations[meta.instrument_id]
            self.assertEqual(observation.ts_event, BASE + 11_000_000_000)
            self.assertFalse(observation.timestamp_regressed)
            row["Volume"] = 11
            feed.on_depth_market_data(row)
            self.assertEqual(enqueue.call_count, 2)
            row["UpdateTime"] = stamp.strftime("%H:%M:%S")
            feed.on_depth_market_data(row)
            self.assertTrue(feed.depth_observations[meta.instrument_id].timestamp_regressed)
            row["UpdateTime"] = later.strftime("%H:%M:%S")
            feed.on_depth_market_data(row)
            self.assertTrue(feed.depth_observations[meta.instrument_id].timestamp_regressed)
        copied = feed.depth_observations
        copied.clear()
        self.assertTrue(feed.depth_observations)
        feed._stop_network_client()
        self.assertFalse(feed.depth_observations)

    def test_role_factor_and_terms_changes_close_gate(self):
        for kind in ("role", "factor", "terms"):
            with self.subTest(kind=kind):
                s = self.session()
                if kind == "terms":
                    # 替换条款对象，模拟数据库原子发布，不能原地修改冻结条款。
                    self.service.instrument_specs["HC"]["main"] = SimpleNamespace(
                        **{**vars(self.specs["HC"]), "tick": Decimal(5)})
                else:
                    self.assignment = SectorRoleAssignment(date(2026, 9, 22), date(2026, 9, 21), BASE, BASE,
                        {"RB": {"main": "rb2705" if kind == "role" else "rb2704"}, "HC": {"main": "hc2704"}},
                        {"RB": {"main": Decimal(3) if kind == "factor" else Decimal(2)}, "HC": {"main": Decimal(1)}})
                with self.assertRaisesRegex(RuntimeError, "变化"):
                    self.frame(s, 0, 3100, 3000)
                self.assertFalse(s.runner.accept_bars)
                s.runner.stop()
                self.setUp_reference_reset()

    def setUp_reference_reset(self):
        self.assignment = SectorRoleAssignment(date(2026, 9, 22), date(2026, 9, 21), BASE, BASE,
            {"RB": {"main": "rb2704"}, "HC": {"main": "hc2704"}},
            {"RB": {"main": Decimal(2)}, "HC": {"main": Decimal(1)}})
        self.service.instrument_specs = {p: {"main": spec} for p, spec in self.specs.items()}

    def test_write_ahead_failure_sends_no_order(self):
        s = self.session(orders=True)
        self.frame(s, 0, 3100, 3000)
        with patch.object(s.manager, "save", side_effect=OSError("fake disk full")):
            with self.assertRaisesRegex(RuntimeError, "fake disk full"):
                self.frame(s, 1, 3130, 2970)
        self.assertFalse(s.transport._api.sent)
        self.assertFalse(s.runner.accept_bars)

    def test_synchronous_rejection_stops_next_leg(self):
        s = self.session(orders=True, api_base=RejectFirstApi)
        self.frame(s, 0, 3100, 3000)
        with self.assertRaises(RuntimeError):
            self.frame(s, 1, 3130, 2970)
        self.assertEqual(len(s.transport._api.sent), 1)
        self.assertIsNotNone(s.client.portfolio_failure)
        self.assertFalse(s.runner.accept_bars)

    def test_lifecycle_accounts_for_all_instruments(self):
        s = self.session(orders=True)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.fill(s, 0)
        self.fill(s, 1)
        self.assertEqual(runtime.CrossSectionLifecycle._expected_gross(s),
                         {"rb2704.SHFE": (1, 0), "hc2704.SHFE": (0, 1)})

    def test_one_filled_leg_cannot_pass_portfolio_acceptance(self):
        s = self.session(orders=True)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.fill(s, 0)
        lifecycle = runtime.CrossSectionLifecycle(s.transport, md_front="tcp://fake:1234", orders=True)
        with self.assertRaisesRegex(RuntimeError, "最新目标尚未完成"):
            lifecycle.verify(s)

    def test_market_tick_aggregation_generates_synchronized_frames(self):
        s = self.session()
        self.raw_ticks(s, BASE + MINUTE + 1, {"RB": 3130, "HC": 2970})
        self.assertEqual(s.strategy.synchronized_frames, 1)
        self.raw_ticks(s, BASE + 2 * MINUTE + 1, {"RB": 3140, "HC": 2960})
        self.assertEqual(s.strategy.synchronized_frames, 2)
        self.assertEqual(s.strategy.rebalances, 1)
        self.assertEqual(s.strategy.last_targets["rb2704.SHFE"], 1)

    def test_other_strategy_cannot_use_shared_client_for_retained_retry(self):
        runner = UnifiedStrategyRunner(RuntimeMode.LIVE, position_manager=PositionManager())
        runner.add_execution_client(RecordingExecutionClient(runtime.CLIENT_ID))
        rb = runtime.PortfolioReferences(self.service).instrument_ids["RB"]
        for name in (runtime.CLIENT_ID, "other"):
            runner.add_strategy(StrategyTemplate(name), data_bindings=(),
                execution_routes=(ExecutionRoute("rb", runtime.CLIENT_ID, rb),))
        with self.assertRaisesRegex(ValueError, "独占"):
            runner.continue_execution_target(runtime.CLIENT_ID, "rb", BASE, trigger_instrument_id=rb)

    def test_partial_fill_dedup_and_working_quantity_prevent_retry(self):
        s = self.session(orders=True, target_notional=Decimal(70000), max_quantity=Decimal(3),
                         max_notional=Decimal(100000))
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        rb_index = next(index for index, row in enumerate(s.transport._api.sent)
                        if row["InstrumentID"] == "rb2704")
        self.assertEqual(s.transport._api.sent[rb_index]["VolumeTotalOriginal"], 2)
        self.fill(s, rb_index, volume=1, duplicate=True)
        rb = s.references.instrument_ids["RB"]
        self.assertEqual(s.strategy.position(str(rb)), 1)
        self.assertEqual(s.runner.position_manager.working_quantity(runtime.CLIENT_ID, rb), 1)
        self.frame(s, 2, 3140, 2960)
        self.assertEqual(len(s.transport._api.sent), 2)
        self.fill(s, rb_index, volume=1, trade_suffix="-second")
        self.assertEqual(s.strategy.position(str(rb)), 2)
        self.assertEqual(s.runner.position_manager.working_quantity(runtime.CLIENT_ID, rb), 0)
        self.assertFalse(s.client.report_errors)

    def five_product_session(self, **options):
        # 仅合成数据装配验证，价格/合约/月不作为真实柜台参数。
        products = ("CU", "PB", "NI", "AU", "AG")
        terms = {"CU": (10, 5), "PB": (5, 5), "NI": (10, 1),
                 "AU": ("0.02", 1000), "AG": (1, 15)}
        self.service.products = products
        self.service.instrument_specs = {p: {"main": SimpleNamespace(symbol=f"{p.lower()}2704",
            product=p, venue="SHFE", currency="CNY", tick=Decimal(str(terms[p][0])),
            multiplier=Decimal(terms[p][1]))} for p in products}
        self.assignment = SectorRoleAssignment(date(2026, 9, 22), date(2026, 9, 21), BASE, BASE,
            {p: {"main": f"{p.lower()}2704"} for p in products},
            {p: {"main": Decimal(1)} for p in products})
        return self.session(**options)

    def test_cu_pb_ni_au_ag_five_product_recording(self):
        s = self.five_product_session(target_notional=Decimal(700000))
        products = s.references.products
        prices = {"CU": 100000, "PB": 20000, "NI": 140000, "AU": 800, "AG": 10000}
        changed = {"CU": 102000, "PB": 20200, "NI": 140000, "AU": 792, "AG": 9800}
        for minute, frame in enumerate((prices, changed)):
            stamp = BASE + (minute + 1) * MINUTE - 1
            self.now = stamp + 1
            s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
            for p in products:
                s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], frame[p], 1,
                    trade_id=f"five-{p}-{minute}", ts_event=self.now, meta=s.references.instrument_meta(p)))
            for p in products[:-1]:
                self.bar(s, p, stamp, frame[p])
            self.assertEqual(s.strategy.synchronized_frames, minute)
            self.bar(s, products[-1], stamp, frame[products[-1]])
        self.assertEqual(s.strategy.synchronized_frames, 2)
        self.assertEqual(s.strategy.last_signal.long_products, ("CU",))
        self.assertEqual(s.strategy.last_signal.short_products, ("AG",))
        self.assertEqual(dict(s.strategy.last_targets), {"cu2704.SHFE": Decimal(1),
            "pb2704.SHFE": Decimal(0), "ni2704.SHFE": Decimal(0),
            "au2704.SHFE": Decimal(0), "ag2704.SHFE": Decimal(-4)})
        self.assertEqual(len(s.client.requests), 1)

    def test_full_groups_cap_one_sends_and_attributes_four_contracts(self):
        s = self.five_product_session(orders=True, target_notional=Decimal(2000000),
            group_fraction=Decimal("0.40"), target_quantity_cap=Decimal(1),
            require_full_groups=True, max_quantity=Decimal(1), max_notional=Decimal(2000000))
        for minute, prices in enumerate((
                {"CU": 100000, "PB": 20000, "NI": 140000, "AU": 800, "AG": 10000},
                {"CU": 102000, "PB": 20200, "NI": 140000, "AU": 792, "AG": 9800})):
            stamp = BASE + (minute + 1) * MINUTE - 1
            self.now = stamp + 1
            s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
            for p in s.references.products:
                s.runner.observe_tick(make_trade_tick(s.references.instrument_ids[p], prices[p], 1,
                    ts_event=self.now, meta=s.references.instrument_meta(p), trade_id=f"{p}-{minute}"))
            for p in s.references.products:
                self.bar(s, p, stamp, prices[p])
        self.assertEqual(dict(s.strategy.last_targets), {"cu2704.SHFE": Decimal(1),
            "pb2704.SHFE": Decimal(1), "ni2704.SHFE": Decimal(0),
            "au2704.SHFE": Decimal(-1), "ag2704.SHFE": Decimal(-1)})
        self.assertEqual(len(s.transport._api.sent), 4)
        for index in range(4):
            self.assertEqual(s.transport._api.sent[index]["VolumeTotalOriginal"], 1)
            self.fill(s, index)
        self.assertEqual(s.strategy.fills_received, 4)
        for key, target in s.strategy.last_targets.items():
            self.assertEqual(s.strategy.position(key), target)
        self.assertFalse(s.client.report_errors)

    def test_full_groups_budget_shortfall_does_not_force_one_lot(self):
        s = self.session(target_notional=Decimal(100), target_quantity_cap=Decimal(1),
            require_full_groups=True)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        request = s.client.requests[-1]
        self.assertTrue(all(q == 0 for q in request.targets.values()))
        self.assertEqual(set(request.metadata["incomplete_group_products"]), {"RB", "HC"})

    def test_full_groups_shortfall_still_closes_existing_positions(self):
        s = self.session(orders=True, target_notional=Decimal(35000),
            target_quantity_cap=Decimal(1), require_full_groups=True, rebalance_interval=1)
        self.frame(s, 0, 3100, 3000)
        self.frame(s, 1, 3130, 2970)
        self.fill(s, 0)
        self.fill(s, 1)
        self.frame(s, 2, 4000, 2970)
        self.assertTrue(all(q == 0 for q in s.strategy.last_targets.values()))
        self.assertEqual(len(s.transport._api.sent), 4)
        self.assertTrue(all(o["CombOffsetFlag"] != "0" for o in s.transport._api.sent[2:]))

    def test_target_quantity_cap_cli_validation(self):
        args = live.parse_args(self.command("--target-quantity-cap", "1", "--require-full-groups"))
        self.assertEqual(args.target_quantity_cap, 1)
        self.assertTrue(args.require_full_groups)
        for value in ("0", "-1", "1.5", "nan", "inf", "2"):
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                live.parse_args(self.command("--target-quantity-cap", value))

    def test_complete_managed_lifecycle_preflight_relogin_fill_shutdown(self):
        args = live.parse_args(self.command("--mode", "simnow", "--enable-orders", "--confirm-simnow",
            "--state-file", str(self.root / "complete.json"), "--report-dir", str(self.root / "reports"),
            "--lookback", "1", "--rebalance-interval", "2", "--target-notional", "35000",
            "--max-notional", "50000", "--seconds", "1", "--query-timeout", "0.05"))
        refs = runtime.PortfolioReferences(self.service)
        owner = self

        class PrimedMd(fixtures.ManualMd):
            def connect(self):
                super().connect()
                owner.raw_ticks(SimpleNamespace(upstream=self, references=refs), BASE,
                                {"RB": 3100, "HC": 3000})

        def inputs(options, context, controller):
            context.references = refs
            return SimpleNamespace(references=refs, upstream=PrimedMd())

        original = CtpTdApiTransport

        def transport(**kwargs):
            return original(**kwargs, td_api_base=fixtures.FlatTdApi)

        env = {"CTP_BROKER_ID": "9999", "CTP_ACCOUNT_ID": "demo", "CTP_PASSWORD": "fake",
            "CTP_TD_ADDRESS": "tcp://fake:1234", "CTP_MD_ADDRESS": "tcp://fake:1234",
            "CTP_APP_ID": "fake-app", "CTP_AUTH_CODE": "fake-auth", "CTP_PRODUCTION_MODE": "true",
            "CTP_TD_FLOW_PATH": str(self.root / "complete-td")}
        with patch.dict("os.environ", env), patch.object(live, "CtpTdApiTransport", side_effect=transport), \
                patch.object(live, "prepare_inputs", side_effect=inputs):
            managed = live.build_runtime(args)
            self.assertIsNone(managed.controller.driver.trading_day)
            monotonic = [0.0]
            minute = [0]
            managed._monotonic = lambda: monotonic[0]
            managed._sleep = lambda seconds: monotonic.__setitem__(0, monotonic[0] + seconds)

            def progress(s):
                minute[0] += 1
                self.raw_ticks(s, BASE + minute[0] * MINUTE + 1,
                               {"RB": 3100 + 30 * minute[0], "HC": 3000 - 30 * minute[0]})
                for index in range(s.strategy.fills_received, len(s.transport._api.sent)):
                    self.fill(s, index, duplicate=True)

            # Session保留原native transport供假回报生成；生产装配不依赖该属性。
            assemble_session = managed.assemble_session

            def assembly(inputs, context):
                s = assemble_session(inputs, context)
                s.transport = managed.controller.transport
                return s

            managed.assemble_session = assembly
            managed.progress = progress
            result = managed.run()
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["orders_submitted"], 2)
            self.assertEqual(result["fills_received"], 2)
            self.assertEqual(result["final_active_orders"], 0)
            self.assertEqual(result["cleanup_errors"], [])
            self.assertEqual(result["final_gross"], {"rb2704.SHFE": (1, 0), "hc2704.SHFE": (0, 1)})
            self.assertEqual(result["fixed_instruments"], args.expected_instruments)
            self.assertTrue(args.state_file.is_file())
            managed.stop()


if __name__ == "__main__":
    unittest.main()
