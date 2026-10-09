"""当前主力EMA在线装配的无网络验证；原生TD传输使用假API。"""

from __future__ import annotations

from datetime import date
from contextlib import ExitStack, nullcontext, redirect_stderr
from decimal import Decimal
from importlib import import_module
from io import StringIO
from pathlib import Path
import json
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from bomber.framework.market.basic.base import MarketDataFeed, make_bar, make_trade_tick
from bomber.framework.market.stream.health import MarketHealthState
from bomber.framework.trader.execution.ctp import CtpNativeTraderDriver, CtpTdApiTransport
from bomber.framework.trader.persistence import ConcurrentStateWriteError, JsonStateStore
from bomber.framework.trader import MarketReferencePriceStore, PositionManager
from bomber.framework.trader import RecordingExecutionClient, RuntimeMode, StrategyTemplate
from bomber.framework.trader.live_roles import FixedRoleLiveRunner
from bomber.framework.trader.assembly import RoleGuard, StrategyBindings, assemble_strategy
from bomber.framework.trader.contracts import DataBinding, ExecutionRoute
from bomber.framework.trader.execution.builders import ExecutionComponents
from bomber.framework.market.basic.base import DataType
from bomber.framework.datahub.sector_roles import SectorDataUnavailable, SectorRoleAssignment
from bomber.framework.trader.contracts import ExecutionRequest
from bomber.framework.trader.execution.ctp import CtpLimitPlanner, CtpPositionLedger
from bomber.framework.trader.runtime.ctp import account_lock

from .run_p4_ctp_td_transport import FakeTdApi

live = import_module("demos.01_main_ema.run_live")
refs = import_module("bomber.framework.dataprep.live_role")
BASE = pd.Timestamp("2026-09-22T09:00:00+08:00").value
MINUTE = 60_000_000_000


class FlatTdApi(FakeTdApi):
    def __init__(self):
        super().__init__()
        self.gross = {}
        self.sent = []
        self.active = False

    def registerFront(self, front):
        # 全程使用假API；地址仅用于验证入口配置，不建立网络连接。
        # 基础夹具只接受fake地址，本入口另覆盖显式第二套前置校验。
        assert front in {"tcp://fake:1234", "tcp://182.254.243.31:40001"}
        self.registered_front = front

    def reqQryInvestorPosition(self, data, reqid):
        self.request_names.append("positions")
        for instrument, amounts in self.gross.items():
            symbol, exchange = instrument.split(".")
            for direction, amount in zip(("2", "3"), amounts):
                if amount:
                    self.onRspQryInvestorPosition({
                        "BrokerID": "9999", "InvestorID": "demo", "InstrumentID": symbol,
                        "ExchangeID": exchange, "PosiDirection": direction, "Position": int(amount),
                    }, {}, reqid, False)
        if not self.drop_position_last:
            self.onRspQryInvestorPosition({}, {}, reqid, True)
        return 0

    def reqQryOrder(self, data, reqid):
        if self.active:
            return super().reqQryOrder(data, reqid)
        self.request_names.append("orders")
        self.onRspQryOrder({}, {}, reqid, True)
        return 0

    def reqOrderInsert(self, data, reqid):
        self.request_names.append("insert")
        self.sent.append(dict(data))
        return 0


class AutoFillTdApi(FlatTdApi):
    def reqOrderInsert(self, data, reqid):
        super().reqOrderInsert(data, reqid)
        self.gross[f"{data['InstrumentID']}.{data['ExchangeID']}"] = (1, 0)
        raw = {"BrokerID": "9999", "InvestorID": "demo", "InstrumentID": data["InstrumentID"],
               "ExchangeID": data["ExchangeID"], "OrderRef": data["OrderRef"],
               "FrontID": 1, "SessionID": 2, "OrderSysID": "FULL-SYS"}
        self.onRtnOrder({**raw, "OrderStatus": "3"})
        self.onRtnTrade({**raw, "TradeID": "FULL-T1", "Volume": data["VolumeTotalOriginal"],
                        "Price": data["LimitPrice"]})
        return 0


class AdoptTdApi(FlatTdApi):
    position_date = "1"
    hedge_flag = "1"
    position_cost = 31000
    missing_cost = False

    def __init__(self):
        super().__init__()
        self.gross = {"rb2704.SHFE": (0, 1)}

    def reqQryInvestorPosition(self, data, reqid):
        self.request_names.append("positions")
        for instrument, amounts in self.gross.items():
            symbol, exchange = instrument.split(".")
            for direction, amount in zip(("2", "3"), amounts):
                if not amount:
                    continue
                row = {"BrokerID": "9999", "InvestorID": "demo", "InstrumentID": symbol,
                    "ExchangeID": exchange, "PosiDirection": direction, "Position": int(amount),
                    "PositionDate": self.position_date, "HedgeFlag": self.hedge_flag,
                    "TodayPosition": int(amount) if self.position_date == "1" else 0}
                if not self.missing_cost:
                    row["PositionCost"] = self.position_cost
                self.onRspQryInvestorPosition(row, {}, reqid, False)
        if not self.drop_position_last:
            self.onRspQryInvestorPosition({}, {}, reqid, True)
        return 0


class ManualMd(MarketDataFeed):
    def __init__(self):
        super().__init__("manual-main-md")
        self.latest_trading_day = "20260922"
        self.latest_receive_monotonic_ns = time.monotonic_ns()
        self.health_snapshot = SimpleNamespace(state=MarketHealthState.READY)

    def connect(self):
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False

    def _on_subscription_added(self, request):
        pass

    def _on_subscription_removed(self, request):
        pass


class LiveTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="main-ema-live-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = BASE
        self.patch_clock = patch("time.time_ns", side_effect=lambda: self.now)
        self.patch_clock.start()
        self.addCleanup(self.patch_clock.stop)
        self.role_path = self.root / "roles.feather"
        self.factor_path = self.root / "factors.feather"
        self.basic_path = self.root / "basic.feather"
        pd.DataFrame([{"trade_date": "2026-09-21", "code": "RB", "main": "rb2703"}]).to_feather(self.role_path)
        self.write_factor()
        pd.DataFrame([{
            "symbol": "rb2704", "code": "RB", "exchangeCD": "XSGE",
            "contMultNum": 10, "minChgPriceNum": 1,
            "listDate": "2026-01-01", "lastTradeDate": "2027-04-30",
        }]).to_feather(self.basic_path)

    def write_factor(self, **changes):
        pd.DataFrame([{"trade_date": "2026-09-22", "code": "RB", "symbol": "rb2704",
                       "pcr_cumfactor": "2", **changes}]).to_feather(self.factor_path)

    def reference(self):
        return refs.FileRoleReferences(
            product="RB", trading_day="20260922", contract_struct=self.role_path,
            factors=self.factor_path, fut_basic=self.basic_path, started_ns=BASE)

    def transport_driver(self, orders=False, api_base=FlatTdApi):
        transport = CtpTdApiTransport(
            client_id=live.CLIENT_ID, account_id="demo", front="tcp://fake:1234",
            broker_id="9999", investor_id="demo", password="fake-password",
            app_id="fake-app", auth_code="fake-auth", td_api_base=api_base,
            flow_path=str(self.root / "td"), timeout_seconds=0.05)
        holder = {}
        driver = CtpNativeTraderDriver(
            live.CLIENT_ID, "demo", transport, enable_simnow_orders=orders,
            max_session_orders=4,
            disconnect_handler=lambda reason: holder["session"].client.mark_disconnected(reason)
                if "session" in holder and orders else None)
        self.addCleanup(driver.stop)
        return transport, driver, holder

    def session(self, orders=False, replay=False, adopt=False):
        transport, driver, holder = self.transport_driver(orders, AdoptTdApi if adopt else FlatTdApi)
        args = SimpleNamespace(mode="simnow" if orders else "recording", product="RB",
            fast=2, slow=3, quantity=Decimal(1), limit_offset_ticks=1,
            max_notional=Decimal(50000), state_file=self.root / "state.json",
            simnow_environment="replay" if replay else "realtime",
            replay_md_trading_day="20260921" if replay else None)
        upstream = ManualMd()
        if replay:
            upstream.latest_trading_day = "20260921"
        if adopt:
            controller = live.MainEmaSessionLifecycle(transport, md_front="tcp://fake:1234", orders=orders,
                adopt_instrument="rb2704.SHFE", expected_position=Decimal(-1))
            controller.driver = driver
            resources = ExitStack()
            self.addCleanup(resources.close)
            context = controller.prepare(resources)
            context.references = self.reference()
            controller.finish_preparation(context)
        s = live.assemble(args, self.reference(), driver, upstream)
        holder["session"] = s
        if adopt:
            controller.start(s, context)
            self.addCleanup(s.runner.stop)
            s.transport = transport
            s.controller = controller
            s.adoption_resources = resources
            return s
        if not orders:
            driver.start(lambda report: None)
        s.runner.start()
        self.addCleanup(s.runner.stop)
        live.assert_flat_account(driver, transport)
        if orders:
            s.client.arm_demo(s.client.DEMO_CONFIRMATION)
        s.runner.accept_bars = True
        s.transport = transport
        return s

    def tick(self, s, minute, price=3100):
        self.now = BASE + minute * MINUTE + 1_000
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        s.upstream._emit_trade_tick(make_trade_tick(
            instrument_id=s.references.instrument_id, price=price, size=1,
            ts_event=self.now - (86_400_000_000_000 if s.replay_md_trading_day else 0),
            ts_init=self.now, meta=s.references.instrument_meta(),
            trade_id=f"MD{minute}"))

    def fill(self, s, index, duplicate=False):
        api = s.transport._api
        fields = api.sent[index]
        raw = {"BrokerID": "9999", "InvestorID": "demo", "InstrumentID": fields["InstrumentID"],
               "ExchangeID": fields["ExchangeID"], "OrderRef": fields["OrderRef"],
               "OrderSysID": f"SYS{index}", "FrontID": 1, "SessionID": 2}
        api.onRtnOrder({**raw, "OrderStatus": "3"})
        trade = {**raw, "TradeID": f"T{index}", "Volume": fields["VolumeTotalOriginal"],
                 "Price": fields["LimitPrice"]}
        api.onRtnTrade(trade)
        if duplicate:
            api.onRtnTrade(trade)

    def test_reference_factor_contract_override_and_real_metadata(self):
        reference = self.reference()
        assignment = reference.snapshot(BASE)
        self.assertEqual(assignment.source_day, date(2026, 9, 21))
        self.assertEqual(str(reference.instrument_id), "rb2704.SHFE")
        self.assertEqual(assignment.factor("RB", "main"), 2)
        self.assertEqual(reference.instrument_meta().multiplier, 10)

    def test_public_fixed_role_runner_supports_non_ema_secondary_strategy(self):
        reference = self.reference()
        assignment = SectorRoleAssignment(date(2026, 9, 22), date(2026, 9, 21), BASE, BASE,
            {"RB": {"secondary": "rb2704"}}, {"RB": {"secondary": Decimal(2)}})
        source = SimpleNamespace(snapshot=lambda stamp: assignment)

        class SecondaryStrategy(StrategyTemplate):
            # 不提供EMA config、data_hub或last_processed_ns，验证显式公共接口。
            def __init__(self):
                super().__init__("other-secondary-strategy")
                self.processed = -1

            def on_bar(self, data_key, bar):
                first = self.processed < 0
                self.processed = bar.ts_event
                if first:
                    self.set_target("other_secondary", Decimal(1), bar.ts_event)

        strategy = SecondaryStrategy()
        client = RecordingExecutionClient("other-client")
        positions = PositionManager()
        positions.apply_account_snapshot(client.client_id, {}, revision=1, ts_event=0)
        runner = FixedRoleLiveRunner(RuntimeMode.LIVE, position_manager=positions)
        instrument = reference.instrument_id
        assemble_strategy(runner, strategy, feeds={"bars": ManualMd()},
            execution=ExecutionComponents(client, positions, MarketReferencePriceStore()),
            bindings=StrategyBindings(
                data=(DataBinding(str(instrument), "bars", instrument, DataType.BAR, "1-MINUTE"),),
                execution=(ExecutionRoute("other_secondary", client.client_id, instrument),),
                role_guard=RoleGuard(source, "RB", "secondary", lambda: strategy.processed)))
        runner.start()
        self.addCleanup(runner.stop)
        for stamp in (BASE, BASE + MINUTE):
            runner.publish("bars", make_bar(reference.instrument_id,
                3100, 3100, 3100, 3100, 1, stamp, meta=reference.instrument_meta()))
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[0].revision, client.requests[1].revision)
        self.assertEqual(client.requests[-1].logical_targets["other_secondary"], Decimal(1))
        self.assertEqual(runner.bound_instrument, reference.instrument_id)

    def test_reference_future_publication_and_missing_current_day(self):
        self.write_factor(available_ns=BASE + MINUTE)
        reference = self.reference()
        with self.assertRaises(refs.SectorDataUnavailable):
            reference.snapshot(BASE)
        self.write_factor(trade_date="2026-09-21")
        with self.assertRaises(refs.SectorDataUnavailable):
            reference.snapshot(BASE + MINUTE)

    def test_deleted_reference_cannot_reuse_cached_snapshot(self):
        reference = self.reference()
        self.factor_path.unlink()
        with self.assertRaises(FileNotFoundError):
            reference.snapshot(BASE)

    def test_cli_orders_need_all_authorization_and_new_state(self):
        with self.assertRaises(SystemExit) as error, redirect_stderr(StringIO()):
            live.parse_args(["--connect", "--product", "RB", "--mode", "simnow"])
        self.assertEqual(error.exception.code, 2)
        path = self.root / "old-state.json"
        path.write_text("old", encoding="utf-8")
        with self.assertRaises(SystemExit) as error, redirect_stderr(StringIO()):
            live.parse_args(["--connect", "--product", "RB", "--mode", "simnow",
                             "--enable-orders", "--confirm-simnow", "--state-file", str(path)])
        self.assertEqual(error.exception.code, 2)
        command = ["--connect", "--product", "RB", "--mode", "simnow",
                   "--enable-orders", "--confirm-simnow", "--state-file", str(self.root / "new-state.json")]
        for extra in ([], ["--expected-source-day", "20261032"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as error, redirect_stderr(StringIO()):
                live.parse_args([*command, *extra])
            self.assertEqual(error.exception.code, 2)

    def test_stale_database_source_day_blocks_order_before_md_start(self):
        from .test_reference_sources import FakeSession
        from bomber.framework.dataprep.sources import DolphinDbReferenceConfig, DolphinDbReferenceSource

        sdk = FakeSession()
        sdk.data["fut_adjustment_factors"]["date"] = "2026-09-21"
        source = DolphinDbReferenceSource(
            DolphinDbReferenceConfig("fake", 8848, "fake", "fake"), session_factory=lambda **kwargs: sdk)
        state_file = self.root / "stale-state.json"
        args = live.parse_args(["--connect", "--product", "RB", "--mode", "simnow",
            "--enable-orders", "--confirm-simnow", "--state-file", str(state_file),
            "--reference-source", "dolphindb", "--expected-source-day", "20260920"])
        with ExitStack() as resources:
            context = SimpleNamespace(trading_day="20260922", resources=resources, references=None)
            lifecycle = SimpleNamespace(md_front="tcp://fake:1234",
                transport=SimpleNamespace(investor_id="demo", password="fake", production_mode=True))
            with patch.dict("os.environ", {"DDB_HOST": "fake", "DDB_USERNAME": "fake",
                                        "DDB_PASSWORD": "fake"}), \
                    patch.object(live.ReferenceSourceFactory, "create_for", return_value=source), \
                    patch.object(live, "CtpLiveDataFeed", side_effect=AssertionError("不应启动MD")):
                with self.assertRaisesRegex(SectorDataUnavailable,
                        "expected=2026-09-20 actual=2026-09-21"):
                    live.prepare_inputs(args, context, lifecycle)
        self.assertTrue(sdk.closed)
        self.assertFalse(state_file.exists())

    def test_net_zero_hedged_account_and_unknown_order_block_start(self):
        transport, driver, _ = self.transport_driver()
        driver.start(lambda report: None)
        transport._api.gross = {"rb2704.SHFE": (1, 1)}
        self.assertEqual(driver.reconcile()["rb2704.SHFE"], 0)
        with self.assertRaisesRegex(RuntimeError, "总仓"):
            live.assert_flat_account(driver, transport)
        transport._api.gross = {}
        transport._api.active = True
        with self.assertRaisesRegex(RuntimeError, "活动订单"):
            live.assert_flat_account(driver, transport)

    def test_query_missing_final_fragment_blocks_start(self):
        transport, driver, _ = self.transport_driver()
        driver.start(lambda report: None)
        transport._api.drop_position_last = True
        with self.assertRaises(TimeoutError):
            live.assert_flat_account(driver, transport)

    def test_ticks_completed_bars_current_strategy_recording_no_orders(self):
        s = self.session()
        for minute in range(3):
            self.tick(s, minute)
        self.assertEqual(s.client.requests, ())
        self.tick(s, 3)
        self.assertEqual(s.strategy.bars_used, 3)
        self.assertEqual(len(s.client.requests), 1)
        self.assertEqual(s.client.requests[0].targets[s.references.instrument_id], 1)
        self.assertEqual(s.transport._api.sent, [])
        self.tick(s, 4)
        self.assertEqual(s.strategy._revision, 1)

    def test_partial_startup_minute_is_excluded(self):
        s = self.session()
        # 假设进程在分钟中段开始观察，第一根收盘Bar只是部分分钟。
        s.runner.first_complete_bar_ns = BASE + 2 * MINUTE - 1
        self.tick(s, 0)
        self.tick(s, 1)
        self.assertEqual(s.strategy.bars_used, 0)
        self.tick(s, 2)
        self.assertEqual(s.strategy.bars_used, 1)

    def test_limit_prices_round_buy_up_sell_down(self):
        reference = self.reference()
        prices = MarketReferencePriceStore()
        prices.update(reference.instrument_id, Decimal("3100.5"), BASE)
        planner = CtpLimitPlanner(CtpPositionLedger(), PositionManager(), prices, Decimal(1), 1)
        for quantity, expected in ((1, 3102), (-1, 3099)):
            request = ExecutionRequest(strategy_id="test", revision=1, client_id=live.CLIENT_ID,
                ts_event=BASE, targets={reference.instrument_id: Decimal(quantity)}, execution_policy="DIRECT")
            self.assertEqual(planner.plan(request)[0].price, expected)

    def test_old_bar_cannot_trade_on_fresh_raw_reception(self):
        s = self.session(orders=True)
        self.tick(s, 0)
        self.now = BASE + 10 * MINUTE
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        tick = make_trade_tick(instrument_id=s.references.instrument_id, price=3100, size=1,
            trade_id="REPLAY", ts_event=BASE + MINUTE + 1000,
            ts_init=self.now, meta=s.references.instrument_meta())
        with self.assertRaisesRegex(RuntimeError, "历史回放"):
            s.upstream._emit_trade_tick(tick)
        self.assertFalse(s.client.is_armed)
        self.assertEqual(s.transport._api.sent, [])

    def test_client_clock_follows_session_and_stale_request_still_disarms(self):
        s = self.session(orders=True)
        self.now += MINUTE
        request = ExecutionRequest(
            strategy_id=s.strategy.strategy_id, revision=1, client_id=live.CLIENT_ID,
            ts_event=self.now, targets={s.references.instrument_id: Decimal(0)},
            execution_policy="DIRECT")
        # 当前空仓目标可进入规划器且无需报单，证明假墙钟已用于受控客户端。
        s.client.submit_targets(request)
        self.assertTrue(s.client.is_armed)
        stale = ExecutionRequest(
            strategy_id=s.strategy.strategy_id, revision=2, client_id=live.CLIENT_ID,
            ts_event=self.now - 120_000_000_001,
            targets={s.references.instrument_id: Decimal(1)}, execution_policy="DIRECT")
        with self.assertRaisesRegex(RuntimeError, "超过120000000000ns"):
            s.client.submit_targets(stale)
        self.assertFalse(s.client.is_armed)
        self.assertEqual(s.transport._api.sent, [])

    def test_actual_td_transport_reverse_and_durable_duplicate_fill(self):
        self.exercise_reverse(replay=False)

    def _check_fresh_ema_first_fill(self, step, expected):
        s = self.session(orders=True)
        key = s.strategy.config.target_key
        self.assertEqual(s.strategy.position(key), 0)
        self.assertEqual(s.runner.position_manager.account_position(live.CLIENT_ID, s.references.instrument_id), 0)
        self.assertEqual(s.runner.position_manager.unassigned_position(live.CLIENT_ID, s.references.instrument_id), 0)
        self.assertEqual(s.strategy.fills_received, 0)
        self.assertIsNone(s.strategy.last_target)
        for minute in range(3):
            self.tick(s, minute, 3100 + minute * step)
        self.assertFalse(s.transport._api.sent)  # 未完成慢线预热，不报单。
        self.tick(s, 3, 3100 + 3 * step)
        api = s.transport._api
        self.assertEqual(s.strategy.last_target, expected)
        self.assertEqual(s.strategy.signal_source, "ema")
        self.assertEqual(len(api.sent), 1)
        self.assertEqual(api.sent[0]["CombOffsetFlag"], "0")
        self.assertEqual(api.sent[0]["Direction"], "0" if expected == 1 else "1")
        self.assertEqual(api.sent[0]["VolumeTotalOriginal"], 1)
        self.assertEqual(s.strategy.position(key), 0)  # 目标和报单都不直接修改实仓。
        self.fill(s, 0, duplicate=True)
        self.assertEqual(s.strategy.fills_received, 1)
        self.assertEqual(s.strategy.last_fill_position, expected)
        self.assertEqual(s.strategy.last_order_position, expected)
        self.assertEqual(s.strategy.position(key), expected)
        self.assertEqual(s.runner.position_manager.account_position(live.CLIENT_ID, s.references.instrument_id), expected)
        self.assertEqual(s.runner.position_manager.unassigned_position(live.CLIENT_ID, s.references.instrument_id), 0)
        self.tick(s, 4, 3100 + 4 * step)
        self.assertEqual(len(api.sent), 1)

    def test_fresh_ema_short_starts_flat_and_opens_from_its_own_fill(self):
        self._check_fresh_ema_first_fill(-10, -1)

    def test_fresh_ema_long_starts_flat_and_opens_from_its_own_fill(self):
        self._check_fresh_ema_first_fill(10, 1)

    def test_fixed_target_cli_has_been_removed(self):
        command = ["--connect", "--product", "CU"]
        for extra in (["--acceptance-target", "LONG"], ["--confirm-acceptance-target"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                live.parse_args(command + extra)

    def test_ema_short_matching_adopted_position_is_reported_as_untriggered(self):
        s = self.session(orders=True, adopt=True)
        for minute in range(4):
            self.tick(s, minute, 3100 - minute * 10)
        self.assertEqual(s.strategy.last_target, -1)
        self.assertEqual(s.strategy.signal_source, "ema")
        self.assertFalse(s.transport._api.sent)
        with self.assertRaisesRegex(RuntimeError, "成交验收未触发：目标=-1 策略仓位=-1"):
            s.controller.verify(s)

    def test_adopted_short_closes_before_long_open_and_duplicate_fill_is_ignored(self):
        for day, offset in (("1", "3"), ("2", "4")):
            with self.subTest(position_date=day), patch.object(AdoptTdApi, "position_date", day):
                s = self.session(orders=True, adopt=True)
                self.assertEqual(s.runner.position_manager.position(s.strategy.strategy_id,
                    s.strategy.config.target_key), -1)
                self.assertEqual(s.ledger.snapshot(s.references.instrument_id).short_total, 1)
                for minute in range(4):
                    self.tick(s, minute, 3100)
                api = s.transport._api
                self.assertEqual(len(api.sent), 1)
                self.assertEqual(api.sent[0]["CombOffsetFlag"], offset)
                self.assertEqual(api.sent[0]["Direction"], "0")
                self.tick(s, 4, 3100)
                self.assertEqual(len(api.sent), 1)
                self.fill(s, 0, duplicate=True)
                api.gross = {}
                self.assertEqual(s.ledger.snapshot(s.references.instrument_id).short_total, 0)
                self.assertEqual(s.strategy.position(s.strategy.config.target_key), 0)
                self.assertEqual(s.strategy.last_fill_position, 0)
                self.assertEqual(s.strategy.fills_received, 1)
                self.assertEqual(s.strategy.last_order_update.status.value, "FILLED")
                self.assertEqual(s.strategy.last_order_position, 0)
                order_updates = s.strategy.order_updates_received
                self.tick(s, 5, 3100)
                self.assertEqual(len(api.sent), 2)
                self.assertEqual(api.sent[1]["CombOffsetFlag"], "0")
                self.assertEqual(api.sent[1]["Direction"], "0")
                self.fill(s, 1)
                api.gross = {"rb2704.SHFE": (1, 0)}
                self.assertEqual(s.ledger.snapshot(s.references.instrument_id).long_total, 1)
                self.assertEqual(s.strategy.position(s.strategy.config.target_key), 1)
                self.assertEqual(s.strategy.last_fill_position, 1)
                self.assertEqual(s.strategy.fills_received, 2)
                self.assertGreater(s.strategy.order_updates_received, order_updates)
                self.assertEqual(s.strategy.last_order_update.status.value, "FILLED")
                self.assertEqual(s.strategy.last_order_position, 1)
                self.assertEqual(s.runner.position_manager.unassigned_position(
                    live.CLIENT_ID, s.references.instrument_id), 0)
                self.assertEqual(s.strategy._revision, 1)
                saved = JsonStateStore(self.root / "state.json").load()
                self.assertEqual(saved.payload["ctp_drivers"][live.CLIENT_ID]["orders"], [])
                s.runner.stop()
                s.driver.stop()
                s.adoption_resources.close()
                (self.root / "state.json").unlink()
                self.now = BASE

    def test_adoption_rejects_wrong_gross_missing_cost_and_active_orders(self):
        from bomber.framework.trader.runtime.ctp_positions import inspect_adopted_position
        transport, driver, _ = self.transport_driver(api_base=AdoptTdApi)
        driver.start(lambda report: None)
        api = transport._api
        for gross in ({}, {"rb2704.SHFE": (1, 1)}, {"rb2704.SHFE": (0, 2)},
                      {"rb2703.SHFE": (0, 1)}):
            api.gross = gross
            with self.subTest(gross=gross), self.assertRaisesRegex(RuntimeError, "预期不符"):
                inspect_adopted_position(driver, transport, "rb2704.SHFE", -1)
        api.gross = {"rb2704.SHFE": (0, 1)}
        api.missing_cost = True
        with self.assertRaisesRegex(RuntimeError, "PositionCost"):
            inspect_adopted_position(driver, transport, "rb2704.SHFE", -1)
        api.missing_cost = False
        for change in ({"position_date": ""}, {"hedge_flag": "2"}, {"position_cost": 0}):
            with patch.multiple(api, **change), self.assertRaises(RuntimeError):
                inspect_adopted_position(driver, transport, "rb2704.SHFE", -1)
        api.active = True
        with self.assertRaisesRegex(RuntimeError, "活动订单"):
            inspect_adopted_position(driver, transport, "rb2704.SHFE", -1)

    def test_adopted_short_same_signal_keeps_position_without_order(self):
        s = self.session(orders=True, adopt=True)
        for minute, price in enumerate((3100, 3090, 3080, 3070, 3060)):
            self.tick(s, minute, price)
        self.assertEqual(s.strategy.last_target, -1)
        self.assertEqual(s.transport._api.sent, [])
        self.assertEqual(s.ledger.snapshot(s.references.instrument_id).short_total, 1)

    def test_adoption_rejects_changing_cost_between_queries(self):
        from bomber.framework.trader.runtime.ctp_positions import inspect_adopted_position
        transport, driver, _ = self.transport_driver(api_base=AdoptTdApi)
        driver.start(lambda report: None)
        first = transport.query_position_details()
        second = tuple({**dict(row), "PositionCost": 32000} for row in first)
        with patch.object(transport, "query_position_details", side_effect=[first, second]):
            with self.assertRaisesRegex(RuntimeError, "查询期间"):
                inspect_adopted_position(driver, transport, "rb2704.SHFE", -1)

    def test_adoption_cli_requires_exact_position_and_two_order_budget(self):
        command = ["--connect", "--product", "CU", "--mode", "simnow", "--enable-orders",
            "--confirm-simnow", "--state-file", str(self.root / "adopt-state.json"),
            "--expected-source-day", "20261008", "--expected-trading-day", "20261009",
            "--adopt-existing-position", "cu2611.SHFE"]
        args = live.parse_args([*command, "--expected-position", "-1", "--max-session-orders", "2"])
        self.assertEqual(args.expected_position, Decimal(-1))
        for extra in ([], ["--expected-position", "0"],
                      ["--expected-position", "-1", "--max-session-orders", "1"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                live.parse_args([*command, *extra])

    def test_replay_reverse_and_duplicate_fill_still_use_actual_td_ledger(self):
        self.exercise_reverse(replay=True)

    def exercise_reverse(self, replay):
        s = self.session(orders=True, replay=replay)
        for minute in range(4):
            self.tick(s, minute, 3100 if minute < 3 else 3000)
        api = s.transport._api
        self.assertEqual(len(api.sent), 1)
        self.assertEqual(api.sent[0]["CombOffsetFlag"], "0")
        self.fill(s, 0, duplicate=True)
        self.assertEqual(s.ledger.snapshot(s.references.instrument_id).long_total, 1)
        generation = s.manager.generation
        self.tick(s, 4, 3000)
        self.assertEqual(api.sent[1]["CombOffsetFlag"], "3")
        self.fill(s, 1)
        self.tick(s, 5, 3000)
        self.assertEqual(api.sent[2]["CombOffsetFlag"], "0")
        self.fill(s, 2)
        self.assertEqual(s.strategy._revision, 2)
        self.assertGreater(s.manager.generation, generation)
        saved = JsonStateStore(self.root / "state.json").load()
        self.assertEqual(saved.payload["ctp_drivers"][live.CLIENT_ID]["orders"], [])

    def test_md_day_mismatch_disarms_before_order(self):
        s = self.session(orders=True)
        s.upstream.latest_trading_day = "20260923"
        self.tick(s, 0)
        with self.assertRaisesRegex(RuntimeError, "交易日不一致"):
            self.tick(s, 1)
        self.assertFalse(s.client.is_armed)
        self.assertEqual(s.transport._api.sent, [])

    def test_stale_raw_market_and_dispatch_failure_propagate(self):
        s = self.session(orders=True)
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns() - 11_000_000_000
        self.assertFalse(live.session_ready(s.driver, s.upstream, "20260922", require_orders=True))
        self.tick(s, 0)
        self.factor_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.tick(s, 1)
        self.assertIsNotNone(s.runner.failure)
        self.assertFalse(s.client.is_armed)
        self.assertEqual(s.transport._api.sent, [])

    def test_write_before_send_failure_does_not_reach_api(self):
        s = self.session(orders=True)
        JsonStateStore(self.root / "state.json").save({"foreign": True}, expected_generation=0)
        for minute in range(3):
            self.tick(s, minute)
        with self.assertRaises(ConcurrentStateWriteError):
            self.tick(s, 3)
        self.assertEqual(s.transport._api.sent, [])
        self.assertFalse(s.client.is_armed)

    def test_reference_contract_change_disarms_fixed_route(self):
        s = self.session(orders=True)
        self.tick(s, 0)
        frame = pd.read_feather(self.basic_path)
        frame.loc[0, "contMultNum"] = 20
        frame.to_feather(self.basic_path)
        with self.assertRaisesRegex(RuntimeError, "条款变更"):
            self.tick(s, 1)
        self.assertFalse(s.client.is_armed)
        self.assertTrue(s.runner.position_manager.is_recovery_required(live.CLIENT_ID))

    def test_complete_recording_entry_precheck_relogin_and_shutdown(self):
        self.complete_entry(orders=False)

    def test_complete_simnow_entry_checkpoint_fill_and_final_gross(self):
        self.complete_entry(orders=True)

    def test_complete_database_recording_entry(self):
        self.complete_entry(orders=False, database=True)

    def test_complete_database_simnow_entry(self):
        self.complete_entry(orders=True, database=True)

    def test_replay_cli_requires_explicit_day_and_preserves_default(self):
        default = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921"])
        self.assertEqual(default.simnow_environment, "realtime")
        for extra in (["--simnow-environment", "replay"],
                      ["--simnow-environment", "replay", "--replay-md-trading-day", "20260230"],
                      ["--replay-md-trading-day", "20260921"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921", *extra])
        args = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921", "--simnow-environment", "replay",
                               "--replay-md-trading-day", "20260921"])
        self.assertEqual(args.replay_md_trading_day, "20260921")

    def test_replay_front_pair_and_production_key_required(self):
        md, td = "tcp://182.254.243.31:40011", "tcp://182.254.243.31:40001"
        live.validate_replay_environment(md, td, True)
        for actual_md, actual_td, production in (
                (md, td, False), (md.replace("40011", "30011"), td, True),
                (md, td.replace("40001", "30001"), True),
                (md.replace("182.254.243.31", "127.0.0.1"), td, True)):
            with self.assertRaises(ValueError):
                live.validate_replay_environment(actual_md, actual_td, production)

    def test_replay_receive_bars_preserve_raw_ticks_and_record_current_targets(self):
        s = self.session(replay=True)
        raw = []
        s.upstream.register_trade_tick_handler(raw.append)
        for minute in range(4):
            self.tick(s, minute)
        self.assertEqual(s.strategy.bars_used, 3)
        self.assertEqual(len(s.client.requests), 1)
        self.assertEqual(s.client.requests[0].ts_event, BASE + 3 * MINUTE - 1)
        self.assertEqual(s.transport._api.sent, [])
        self.assertEqual(raw[0].ts_event, BASE - 86_400_000_000_000 + 1000)
        self.assertEqual(raw[0].ts_init, BASE + 1000)
        timing = s.bar_feed.timing_snapshot
        self.assertEqual(timing["ticks_used"], 4)
        self.assertEqual(timing["first_source_event_ns"], raw[0].ts_event)
        self.assertEqual(timing["last_receive_ns"], self.now)

    def test_replay_partial_start_minute_still_excluded(self):
        s = self.session(replay=True)
        s.runner.first_complete_bar_ns = BASE + 2 * MINUTE - 1
        self.tick(s, 0)
        self.tick(s, 1)
        self.assertEqual(s.strategy.bars_used, 0)
        self.tick(s, 2)
        self.assertEqual(s.strategy.bars_used, 1)

    def test_ready_wait_crossing_minute_excludes_new_partial_minute(self):
        s = self.session(replay=True)
        self.now = BASE + MINUTE + 30_000_000_000
        s.runner.begin_bars()
        self.tick(s, 1)
        self.tick(s, 2)
        self.assertEqual(s.strategy.bars_used, 0)
        self.tick(s, 3)
        self.assertEqual(s.strategy.bars_used, 1)

    def test_replay_invalid_or_backwards_receive_time_rejected(self):
        s = self.session(replay=True)
        self.tick(s, 0)
        def raw(received):
            return make_trade_tick(instrument_id=s.references.instrument_id,
                price=3100, size=1, trade_id=f"BAD{received}",
                ts_event=BASE - 86_400_000_000_000, ts_init=received,
                meta=s.references.instrument_meta())
        for received in (0, self.now + 1, self.now - 10_000_000_001):
            with self.assertRaisesRegex(RuntimeError, "接收时间"):
                s.upstream._emit_trade_tick(raw(received))
        with self.assertRaisesRegex(RuntimeError, "倒退"):
            s.upstream._emit_trade_tick(raw(self.now - 1))
        self.assertEqual(s.strategy.bars_used, 0)
        self.assertEqual(s.transport._api.sent, [])

    def test_replay_md_switch_disarms_before_signal_order(self):
        s = self.session(orders=True, replay=True)
        for minute in range(3):
            self.tick(s, minute)
        s.upstream.latest_trading_day = "20260922"
        with self.assertRaisesRegex(RuntimeError, "交易日不一致"):
            self.tick(s, 3)
        self.assertFalse(s.client.is_armed)
        self.assertEqual(s.transport._api.sent, [])

    def test_replay_stale_execution_request_still_disarms(self):
        s = self.session(orders=True, replay=True)
        old = ExecutionRequest(strategy_id=s.strategy.strategy_id, revision=1,
            client_id=live.CLIENT_ID, ts_event=BASE - 86_400_000_000_000,
            targets={s.references.instrument_id: Decimal(1)}, execution_policy="DIRECT")
        with self.assertRaisesRegex(RuntimeError, "超过120000000000ns"):
            s.client.submit_targets(old)
        self.assertFalse(s.client.is_armed)
        self.assertEqual(s.transport._api.sent, [])

    def test_replay_readiness_still_checks_td_health_and_reception(self):
        driver = SimpleNamespace(trading_day="20260922", is_simnow_session=True)
        upstream = SimpleNamespace(latest_trading_day="20260921",
            latest_receive_monotonic_ns=100, health_snapshot=SimpleNamespace(state=MarketHealthState.READY))
        def ready():
            return live.session_ready(driver, upstream, "20260922", require_orders=True,
                now_ns=200, replay_md_trading_day="20260921")
        self.assertTrue(ready())
        self.assertFalse(live.session_ready(driver, upstream, "20260922", now_ns=200))
        driver.trading_day = "20260923"
        self.assertFalse(ready())
        driver.trading_day = "20260922"
        upstream.health_snapshot.state = MarketHealthState.DEGRADED
        self.assertFalse(ready())
        upstream.health_snapshot.state = MarketHealthState.READY
        upstream.latest_receive_monotonic_ns = -11_000_000_000
        self.assertFalse(ready())
        upstream.latest_receive_monotonic_ns = 100
        driver.is_simnow_session = False
        self.assertFalse(ready())

    def test_complete_replay_database_recording_entry(self):
        self.complete_entry(orders=False, database=True, replay=True)

    def test_complete_replay_database_simnow_entry(self):
        self.complete_entry(orders=True, database=True, replay=True)

    def test_reference_date_cli_defaults_and_invalid_policy(self):
        file = live.parse_args(["--connect", "--product", "RB", "--reference-source", "file", "--allow-file-reference-test"])
        database = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921"])
        self.assertEqual(database.reference_source, "dolphindb")
        self.assertEqual((file.factor_date_basis, file.factor_availability), ("trading", "aligned"))
        self.assertEqual((database.factor_date_basis, database.factor_availability), ("source", "source-day-end"))
        aligned_database = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921", "--reference-source", "dolphindb",
            "--factor-date-basis", "trading"])
        self.assertEqual(aligned_database.factor_availability, "aligned")
        night_database = live.parse_args(["--connect", "--product", "CU", "--reference-source", "dolphindb",
            "--factor-availability", "observed-on-read", "--expected-source-day", "20261008"])
        self.assertEqual((night_database.product, night_database.factor_availability), ("CU", "observed-on-read"))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921", "--factor-date-basis", "source",
                "--factor-availability", "aligned"])
        self.assertEqual(error.exception.code, 2)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            live.parse_args(["--connect", "--product", "RB", "--reference-source", "file", "--allow-file-reference-test", "--factor-date-basis", "source",
                "--factor-availability", "observed-on-read"])
        self.assertEqual(error.exception.code, 2)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921", "--reference-source", "dolphindb",
                "--simnow-environment", "replay", "--replay-md-trading-day", "20260930",
                "--factor-availability", "observed-on-read"])
        self.assertEqual(error.exception.code, 2)

    def test_ctp_poll_readiness_failure_reports_age_and_days(self):
        upstream = SimpleNamespace(latest_receive_monotonic_ns=1_000_000_000,
            latest_trading_day="20261009",
            health_snapshot=SimpleNamespace(state=MarketHealthState.READY, reason="ready"))
        driver = SimpleNamespace(trading_day="20261009", is_simnow_session=True)
        controller = SimpleNamespace(lost=[], driver=driver, replay_md_trading_day=None)
        runner = SimpleNamespace(failure=None, expected_day="20261009",
            session_ready=lambda: live.session_ready(driver, upstream, "20261009"))
        session = SimpleNamespace(upstream=upstream, runner=runner)
        with patch("time.monotonic_ns", return_value=12_000_000_000):
            with self.assertRaises(RuntimeError) as caught:
                live.CtpSessionLifecycle.poll(controller, session)
        message = str(caught.exception)
        self.assertIn("receive_age_seconds=11.0", message)
        self.assertIn("expected_MD=20261009", message)
        self.assertIn("expected_TD=20261009", message)
        self.assertIn("reason=ready", message)

    def test_ctp_snapshot_preserves_recent_execution_audit(self):
        s = self.session(orders=True)
        controller = live.CtpSessionLifecycle(s.transport, md_front="tcp://fake:1234", orders=True)
        controller.driver = s.driver
        context = SimpleNamespace(references=s.references)
        audit = controller.snapshot(s, context)["execution_audit"]
        self.assertTrue(any(item["action"] == "DEMO_ARMED" for item in audit))
        for index in range(35):
            s.client.disarm(f"diagnostic-{index}")
        audit = controller.snapshot(s, context)["execution_audit"]
        self.assertEqual(len(audit), 30)
        self.assertEqual(audit[-1]["action"], "DISARMED")
        self.assertEqual(audit[-1]["detail"], "diagnostic-34")
        self.assertNotIn("fake-password", str(audit))

    def test_slow_reference_refresh_blocks_bar_before_strategy_decision(self):
        s = self.session(orders=True)
        self.now = BASE + 2 * MINUTE
        s.upstream.latest_receive_monotonic_ns = time.monotonic_ns()
        bar = make_bar(s.references.instrument_id, 3100, 3100, 3100, 3100, 1,
            self.now - 1, meta=s.references.instrument_meta())
        def delay():
            self.now += 121_000_000_000
        with patch.object(s.references, "refresh", side_effect=delay):
            with self.assertRaisesRegex(RuntimeError, "刷新后分钟Bar已过期"):
                s.runner.publish("ctp-bars", bar)
        self.assertEqual(s.strategy.bars_used, 0)
        self.assertFalse(s.client.is_armed)
        self.assertEqual(s.transport._api.sent, [])

    def test_runtime_construction_does_not_connect_or_open_database(self):
        args = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921", "--reference-source", "dolphindb",
            "--report-dir", str(self.root / "reports")])
        environment = {"CTP_BROKER_ID": "9999", "CTP_ACCOUNT_ID": "demo", "CTP_PASSWORD": "fake",
            "CTP_TD_ADDRESS": "tcp://fake:1234", "CTP_MD_ADDRESS": "tcp://fake:1234"}
        with patch.dict("os.environ", environment, clear=True), \
                patch.object(CtpTdApiTransport, "connect") as connect, \
                patch.object(live.ReferenceSourceFactory, "create_for") as create:
            runtime = live.build_runtime(args)
            self.assertIsNone(runtime.report.path)
            runtime.stop()
            connect.assert_not_called()
            create.assert_not_called()
        self.assertFalse((self.root / "reports").exists())

    def test_database_input_failure_closes_source_and_preflight_td(self):
        from .test_reference_sources import FakeSession
        from bomber.framework.dataprep.sources import DolphinDbReferenceConfig, DolphinDbReferenceSource
        from bomber.framework.datahub.sector_roles import SectorDataUnavailable
        args = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921", "--reference-source", "dolphindb",
            "--query-timeout", "0.05", "--report-dir", str(self.root / "reports")])
        sdk = FakeSession()
        sdk.data["fut_adjustment_factors"] = sdk.data["fut_adjustment_factors"].iloc[:0]
        source = DolphinDbReferenceSource(DolphinDbReferenceConfig("fake", 8848, "fake", "fake"),
            session_factory=lambda **kwargs: sdk)
        transports = []

        def td_factory(**kwargs):
            transport = CtpTdApiTransport(**kwargs, td_api_base=FlatTdApi)
            transports.append(transport)
            return transport

        environment = {"CTP_BROKER_ID": "9999", "CTP_ACCOUNT_ID": "demo", "CTP_PASSWORD": "fake",
            "CTP_TD_ADDRESS": "tcp://fake:1234", "CTP_MD_ADDRESS": "tcp://fake:1234",
            "CTP_TD_FLOW_PATH": str(self.root / "td-fail"),
            "DDB_HOST": "fake", "DDB_USERNAME": "fake", "DDB_PASSWORD": "fake"}
        with patch.dict("os.environ", environment, clear=True), \
                patch.object(live, "CtpTdApiTransport", side_effect=td_factory), \
                patch.object(live.ReferenceSourceFactory, "create_for", return_value=source):
            with self.assertRaises(SectorDataUnavailable):
                live.run_session(args)
        summary = json.loads(next((self.root / "reports").glob("*/summary.json")).read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["orders_submitted"], 0)
        self.assertEqual(summary["cleanup_errors"], [])
        self.assertTrue(sdk.closed)
        self.assertIsNone(transports[0]._api)

    def test_runtime_holds_shared_and_legacy_account_locks_before_td_start(self):
        args = live.parse_args(["--connect", "--product", "RB", "--expected-source-day", "20260921"])
        environment = {"CTP_BROKER_ID": "9999", "CTP_ACCOUNT_ID": "demo", "CTP_PASSWORD": "fake",
            "CTP_TD_ADDRESS": "tcp://fake:1234", "CTP_MD_ADDRESS": "tcp://fake:1234",
            "CTP_TD_FLOW_PATH": str(self.root / "td-lock")}
        with patch.dict("os.environ", environment, clear=True), \
                patch.object(live, "CtpTdApiTransport", side_effect=lambda **kwargs:
                    CtpTdApiTransport(**kwargs, td_api_base=FlatTdApi)):
            first, second = live.build_runtime(args), live.build_runtime(args)
        with ExitStack() as resources:
            try:
                first.controller.prepare(resources)
                for namespace in ("bomber-ctp", "bomber-main-ema"):
                    with self.subTest(namespace=namespace), self.assertRaisesRegex(RuntimeError, "已有进程"):
                        with account_lock("demo", namespace=namespace):
                            self.fail("已持有的账户锁不应再次获取")
                with ExitStack() as other_resources, \
                        patch.object(second.controller.driver, "start") as start:
                    with self.assertRaisesRegex(RuntimeError, "已有进程"):
                        second.controller.prepare(other_resources)
                    start.assert_not_called()
            finally:
                first.controller.driver.stop()
        # 两个命名空间均释放，停止后的锁文件仍可用于下一次进程互斥。
        for namespace in ("bomber-ctp", "bomber-main-ema"):
            with account_lock("demo", namespace=namespace):
                pass

    def complete_entry(self, orders, database=False, replay=False):
        args = live.parse_args(["--connect", "--product", "RB", "--fast", "2", "--slow", "3",
            "--expected-source-day", "20260921",
            "--seconds", "0.01", "--query-timeout", "0.05", "--contract-struct", str(self.role_path),
            "--fut-basic", str(self.basic_path), "--factors", str(self.factor_path),
            "--report-dir", str(self.root / "reports"), *(
                ["--mode", "simnow", "--enable-orders", "--confirm-simnow",
                 "--state-file", str(self.root / "complete-state.json")] if orders else []), *(
                ["--reference-source", "dolphindb"] if database else
                ["--reference-source", "file", "--allow-file-reference-test"]), *(
                ["--simnow-environment", "replay", "--replay-md-trading-day", "20260921"]
                if replay else [])])
        upstream = ManualMd()
        if replay:
            upstream.latest_trading_day = "20260921"
        transports = []
        database_source = None
        if database:
            from .test_reference_sources import FakeSession
            from bomber.framework.dataprep.sources import DolphinDbReferenceConfig, DolphinDbReferenceSource
            sdk = FakeSession()
            sdk.data["fut_adjustment_factors"]["date"] = "2026-09-21"
            database_source = DolphinDbReferenceSource(
                DolphinDbReferenceConfig("fake", 8848, "fake", "fake"), session_factory=lambda **kwargs: sdk)

        def td_factory(**kwargs):
            transport = CtpTdApiTransport(**kwargs, td_api_base=AutoFillTdApi if orders else FlatTdApi)
            transports.append(transport)
            return transport

        pumped = False
        original_sleep = time.sleep

        def pump(seconds):
            nonlocal pumped
            if not pumped:
                pumped = True
                instrument = next(iter(upstream.get_subscribed_instruments()))
                for minute in range(4):
                    self.now = BASE + minute * MINUTE + 1000
                    upstream.latest_receive_monotonic_ns = time.monotonic_ns()
                    upstream._emit_trade_tick(make_trade_tick(
                        instrument_id=instrument, price=3100, size=1, trade_id=f"FULL{minute}",
                        ts_event=self.now - (86_400_000_000_000 if replay else 0),
                        ts_init=self.now, meta=upstream.get_instrument_meta(instrument)))
            original_sleep(0.02)

        environment = {"CTP_BROKER_ID": "9999", "CTP_ACCOUNT_ID": "demo",
            "CTP_PASSWORD": "fake", "CTP_APP_ID": "fake", "CTP_AUTH_CODE": "fake",
            "CTP_TD_ADDRESS": "tcp://fake:1234", "CTP_MD_ADDRESS": "tcp://fake:1234",
            "CTP_TD_FLOW_PATH": str(self.root / "full-td")}
        if database:
            environment.update(DDB_HOST="fake", DDB_USERNAME="fake", DDB_PASSWORD="fake")
        if replay:
            environment.update(CTP_MD_ADDRESS="tcp://182.254.243.31:40011",
                CTP_TD_ADDRESS="tcp://182.254.243.31:40001", CTP_PRODUCTION_MODE="true")
        with patch.dict("os.environ", environment, clear=True), \
                patch.object(live, "CtpTdApiTransport", side_effect=td_factory), \
                patch.object(live, "CtpLiveDataFeed", return_value=upstream), \
                patch("time.sleep", side_effect=pump), \
                (patch.object(live.ReferenceSourceFactory, "create_for", return_value=database_source)
                 if database else nullcontext()), \
                (patch.object(live, "resolve_paths", side_effect=AssertionError("数据库模式不应读取文件路径"))
                 if database else nullcontext()):
            live.run_session(args)
        report = next((self.root / "reports").glob("*/summary.json"))
        summary = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["simnow_environment"], "replay" if replay else "realtime")
        self.assertEqual(summary["bar_time_basis"], "receive" if replay else "event")
        if replay:
            self.assertEqual(summary["replay_md_trading_day"], "20260921")
            self.assertEqual(summary["observed_md_trading_day"], "20260921")
            self.assertEqual(summary["observed_td_trading_day"], "20260922")
            self.assertEqual(summary["timing"]["ticks_used"], 4)
            self.assertEqual(summary["timing"]["first_source_event_ns"], BASE - 86_400_000_000_000 + 1000)
        if database:
            self.assertTrue(sdk.closed)
            self.assertEqual(summary["reference_source"], "dolphindb")
            self.assertEqual(summary["reference_files"], [])
            self.assertIn("fingerprint", summary["reference_manifest"])
            self.assertEqual(summary["factor_date_basis"], "source")
            self.assertEqual(summary["factor_date"], "2026-09-21")
            self.assertEqual(summary["reference_manifest"]["factor_publication_policy"], "source-day-end")
        self.assertEqual(summary["orders_submitted"], 1 if orders else 0)
        self.assertEqual(summary["bars_used"], 3)
        self.assertEqual(summary["final_active_orders"], 0)
        self.assertEqual(summary["final_gross"], {"rb2704.SHFE": ["1", "0"]} if orders else {})
        self.assertFalse(upstream.is_connected)
        self.assertIsNone(transports[0]._api)
        if orders:
            saved = JsonStateStore(self.root / "complete-state.json").load()
            self.assertEqual(saved.payload["ctp_drivers"][live.CLIENT_ID]["orders"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
