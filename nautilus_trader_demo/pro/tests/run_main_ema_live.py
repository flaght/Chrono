"""当前主力EMA在线装配的无网络验证；原生TD传输使用假API。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from importlib import import_module
from pathlib import Path
import json
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from bomber.framework.market.basic.base import MarketDataFeed, make_trade_tick
from bomber.framework.market.stream.health import MarketHealthState
from bomber.framework.trader.execution.ctp import CtpNativeTraderDriver, CtpTdApiTransport
from bomber.framework.trader.persistence import ConcurrentStateWriteError, JsonStateStore
from bomber.framework.trader import MarketReferencePriceStore, PositionManager
from bomber.framework.trader.contracts import ExecutionRequest
from bomber.framework.trader.execution.ctp import CtpLimitPlanner, CtpPositionLedger

from .run_p4_ctp_td_transport import FakeTdApi

live = import_module("demos.01_main_ema.run_live")
refs = import_module("demos.01_main_ema.live_references")
BASE = pd.Timestamp("2026-09-22T09:00:00+08:00").value
MINUTE = 60_000_000_000


class FlatTdApi(FakeTdApi):
    def __init__(self):
        super().__init__()
        self.gross = {}
        self.sent = []
        self.active = False

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
        return refs.LiveMainReferences(
            product="RB", trading_day="20260922", contract_struct=self.role_path,
            factors=self.factor_path, fut_basic=self.basic_path, started_ns=BASE)

    def transport_driver(self, orders=False):
        transport = CtpTdApiTransport(
            client_id=live.CLIENT_ID, account_id="demo", front="tcp://fake:1234",
            broker_id="9999", investor_id="demo", password="fake-password",
            app_id="fake-app", auth_code="fake-auth", td_api_base=FlatTdApi,
            flow_path=str(self.root / "td"), timeout_seconds=0.05)
        holder = {}
        driver = CtpNativeTraderDriver(
            live.CLIENT_ID, "demo", transport, enable_simnow_orders=orders,
            max_session_orders=4,
            disconnect_handler=lambda reason: holder["session"].client.mark_disconnected(reason)
                if "session" in holder and orders else None)
        self.addCleanup(driver.stop)
        return transport, driver, holder

    def session(self, orders=False):
        transport, driver, holder = self.transport_driver(orders)
        args = SimpleNamespace(mode="simnow" if orders else "recording", product="RB",
            fast=2, slow=3, quantity=Decimal(1), limit_offset_ticks=1,
            max_notional=Decimal(50000), state_file=self.root / "state.json")
        s = live.assemble(args, self.reference(), driver, ManualMd())
        holder["session"] = s
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
            ts_event=self.now, ts_init=self.now, meta=s.references.instrument_meta(),
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
        with self.assertRaises(SystemExit):
            live.parse_args(["--connect", "--product", "RB", "--mode", "simnow"])
        path = self.root / "old-state.json"
        path.write_text("old", encoding="utf-8")
        with self.assertRaises(SystemExit):
            live.parse_args(["--connect", "--product", "RB", "--mode", "simnow",
                             "--enable-orders", "--confirm-simnow", "--state-file", str(path)])

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

    def test_actual_td_transport_reverse_and_durable_duplicate_fill(self):
        s = self.session(orders=True)
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

    def complete_entry(self, orders):
        args = live.parse_args(["--connect", "--product", "RB", "--fast", "2", "--slow", "3",
            "--seconds", "0.01", "--query-timeout", "0.05", "--contract-struct", str(self.role_path),
            "--fut-basic", str(self.basic_path), "--factors", str(self.factor_path),
            "--report-dir", str(self.root / "reports"), *(
                ["--mode", "simnow", "--enable-orders", "--confirm-simnow",
                 "--state-file", str(self.root / "complete-state.json")] if orders else [])])
        upstream = ManualMd()
        transports = []

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
                        ts_event=self.now, ts_init=self.now, meta=upstream.get_instrument_meta(instrument)))
            original_sleep(0.02)

        environment = {"CTP_BROKER_ID": "9999", "CTP_ACCOUNT_ID": "demo",
            "CTP_PASSWORD": "fake", "CTP_APP_ID": "fake", "CTP_AUTH_CODE": "fake",
            "CTP_TD_ADDRESS": "tcp://fake:1234", "CTP_MD_ADDRESS": "tcp://fake:1234",
            "CTP_TD_FLOW_PATH": str(self.root / "full-td")}
        with patch.dict("os.environ", environment, clear=True), \
                patch.object(live, "CtpTdApiTransport", side_effect=td_factory), \
                patch.object(live, "CtpLiveDataFeed", return_value=upstream), \
                patch("time.sleep", side_effect=pump):
            live.run_session(args)
        report = next((self.root / "reports").glob("*/summary.json"))
        summary = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed")
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
