"""正式IM Recording装配/生命周期与历史实时衔接；假柜台，不连接网络。"""
from contextlib import redirect_stderr
from datetime import datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import json
import unittest

import pandas as pd

from bomber.framework.trader.runtime.trading_sessions import IndexSessions, load_sessions, SHANGHAI
from tests import run_main_ema_live as live_fixture

ManualMd, BASE, MINUTE = live_fixture.ManualMd, live_fixture.BASE, live_fixture.MINUTE
live, refs = live_fixture.live, live_fixture.refs

ROOT = Path(__file__).resolve().parents[1]
CALENDAR = ROOT / "demos/01_main_ema/cffex_im_20260921_22.json"
OPEN = BASE + 30 * MINUTE


class ImRuntimeTests(unittest.TestCase):
    def fixture(self):
        fixture = live_fixture.LiveTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        pd.DataFrame([{"trade_date": "2026-09-21", "code": "IM", "main": "IM2610"}]).to_feather(fixture.role_path)
        pd.DataFrame([{"trade_date": "2026-09-22", "code": "IM", "symbol": "IM2610",
            "pcr_cumfactor": "1.456591"}]).to_feather(fixture.factor_path)
        pd.DataFrame([{"symbol": "IM2610", "code": "IM", "exchangeCD": "CCFX",
            "contMultNum": 200, "minChgPriceNum": 0.2,
            "listDate": "2026-01-01", "lastTradeDate": "2026-10-16"}]).to_feather(fixture.basic_path)
        fixture.now = OPEN + 2 * MINUTE
        references = refs.FileRoleReferences(product="IM", trading_day="20260922",
            contract_struct=fixture.role_path, factors=fixture.factor_path, fut_basic=fixture.basic_path,
            started_ns=OPEN, allowed_venues=("CFFEX",))
        self.assertEqual(str(references.instrument_id), "IM2610.CFFEX",
            f"CFFEX标识应为大写；实际加载参考模块={refs.__file__}，"
            f"spec.symbol={references.spec.symbol!r} venue={references.spec.venue!r}；"
            "请核对公共live_role.py与PF0916部署摘要")
        return fixture, references

    def args(self, fixture, **changes):
        path = fixture.root / "im-history.jsonl"
        factor = Decimal("1.456591")
        path.write_text("\n".join(json.dumps({"instrument_id": "IM2610.CFFEX",
            "ts_event": OPEN + (index + 1) * MINUTE - 1,
            "adjusted_close": str(Decimal("7760") * factor)}) for index in range(2)))
        values = dict(mode="recording", product="IM", expected_instrument="IM2610.CFFEX",
            fast=1, slow=2, quantity=Decimal(1), limit_offset_ticks=1,
            max_notional=Decimal(2000000), state_file=None, history_minutes=2,
            history_config=None, history_db="dolphindb", history_file=path, history_missing_policy="fail",
            history_stage="preopen", reference_timeout=15, trading_calendar=CALENDAR)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_calendar_factory_im_sessions_lunch_and_previous_day(self):
        calendar = load_sessions(CALENDAR, "IM")
        self.assertIsInstance(calendar, IndexSessions)
        def moment(value): return datetime.fromisoformat("2026-09-22T" + value).replace(tzinfo=SHANGHAI)
        self.assertIsNone(calendar.window(moment("09:29:59")))
        self.assertIsNotNone(calendar.window(moment("09:30:00")))
        self.assertIsNone(calendar.window(moment("11:30:00")))
        self.assertIsNotNone(calendar.window(moment("13:00:00")))
        self.assertIsNone(calendar.window(moment("15:00:00")))
        self.assertEqual(calendar.previous_trading_day("20260922"), "20260921")
        cutoff = int(moment("15:00:00").timestamp()) * 10**9
        self.assertEqual(len(calendar.last_minutes(cutoff, 240)), 240)
        previous_close = int(datetime(2026, 9, 21, 15, tzinfo=SHANGHAI).timestamp()) * 10**9 - 1
        self.assertEqual(calendar.last_minutes(OPEN, 1)[0], previous_close)

    def test_calendar_unknown_day_and_insufficient_coverage_fail(self):
        calendar = IndexSessions(CALENDAR)
        with self.assertRaises(RuntimeError):
            calendar.window(datetime(2026, 9, 23, 9, 30, tzinfo=SHANGHAI))
        with self.assertRaises(RuntimeError): calendar.last_minutes(OPEN, 241)
        with self.assertRaises(ValueError): calendar.window(datetime(2026, 9, 22, 9, 30))

    def test_cli_requires_expected_im_contract_and_correct_history_calendar(self):
        base = ["--connect", "--product", "IM", "--expected-source-day", "20260921"]
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit): live.parse_args(base)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            live.parse_args([*base, "--expected-instrument", "IM2610.CFFEX", "--history-minutes", "240"])
        args = live.parse_args([*base, "--expected-instrument", "IM2610.CFFEX", "--history-minutes", "240",
            "--trading-calendar", str(CALENDAR)])
        self.assertEqual(args.mode, "recording")

    def test_formal_recording_lifecycle_preloads_history_then_records_only_new_minute(self):
        fixture, references = self.fixture()
        transport, driver, _ = fixture.transport_driver()
        driver.start(lambda report: None)
        session = live.assemble(self.args(fixture), references, driver, ManualMd())
        fixture.addCleanup(session.runner.stop)
        controller = live.MainEmaSessionLifecycle(transport, md_front="tcp://fake:1234", orders=False)
        controller.driver = driver
        controller.start(session, SimpleNamespace(trading_day="20260922"))
        self.assertEqual(str(session.references.instrument_id), "IM2610.CFFEX")
        self.assertEqual(session.strategy.bars_used, 2)
        self.assertEqual(session.strategy._revision, 0)
        self.assertEqual(session.client.requests, ())
        fixture.tick(session, 32, price=7760)
        fixture.tick(session, 33, price=7760)
        self.assertEqual(session.strategy.bars_used, 3)
        self.assertEqual(len(session.client.requests), 1)
        self.assertEqual(transport._api.sent, [])
        self.assertEqual(session.strategy.fills_received, 0)

    def test_assembly_rejects_im_orders_and_wrong_expected_contract(self):
        fixture, references = self.fixture()
        transport, driver, _ = fixture.transport_driver()
        with self.assertRaises(ValueError):
            live.assemble(self.args(fixture, mode="simnow"), references, driver, ManualMd())
        with self.assertRaises(ValueError):
            live.assemble(self.args(fixture, expected_instrument="IM2611.CFFEX"), references, driver, ManualMd())

    def test_formal_history_gap_keeps_recording_bar_gate_closed(self):
        from bomber.framework.dataprep.history import HistoryUnavailable
        fixture, references = self.fixture()
        transport, driver, _ = fixture.transport_driver()
        driver.start(lambda report: None)
        session = live.assemble(self.args(fixture, history_minutes=3), references, driver, ManualMd())
        fixture.addCleanup(session.runner.stop)
        controller = live.MainEmaSessionLifecycle(transport, md_front="tcp://fake:1234", orders=False)
        controller.driver = driver
        with self.assertRaises(HistoryUnavailable):
            controller.start(session, SimpleNamespace(trading_day="20260922"))
        self.assertFalse(session.runner.accept_bars)
        self.assertEqual(session.strategy.bars_used, 0)
        self.assertEqual(session.client.requests, ())
        self.assertEqual(transport._api.sent, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
