"""独立一手平仓工具与原生持仓明细的假TD验收；不读真实配置、不联网。"""

from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bomber.framework.trader.execution.ctp import CtpTdApiTransport
from bomber.framework.trader.runtime.ctp import account_lock
from scripts.integration.ctp import close_existing as entry
from .run_p4_ctp_td_transport import FakeTdApi


class CloseTdApi(FakeTdApi):
    initial_gross = {"rb2701.SHFE": (0, 1)}
    position_day = "1"
    hedge_flag = "1"
    has_active = False
    fill = True
    foreign = False
    drop_last = False
    change_on_details = False

    def __init__(self):
        super().__init__()
        self.gross = dict(self.initial_gross)
        self.sent = []
        self.position_queries = 0
        self.created.append(self)

    def reqQryInvestorPosition(self, data, reqid):
        self.position_queries += 1
        if self.change_on_details and self.position_queries == 2:
            self.gross = {"rb2701.SHFE": (0, 2)}
        for key, amounts in self.gross.items():
            symbol, exchange = key.split(".")
            for direction, quantity in zip(("2", "3"), amounts):
                if not quantity:
                    continue
                row = {"BrokerID": "9999", "InvestorID": "wrong" if self.foreign else "demo",
                    "InstrumentID": symbol, "ExchangeID": exchange, "PosiDirection": direction,
                    "Position": quantity, "TodayPosition": quantity if self.position_day == "1" else 0,
                    "YdPosition": quantity if self.position_day == "2" else 0}
                if self.position_day is not None:
                    row["PositionDate"] = self.position_day
                if self.hedge_flag is not None:
                    row["HedgeFlag"] = self.hedge_flag
                self.onRspQryInvestorPosition(row, {}, reqid, False)
        if not self.drop_last:
            self.onRspQryInvestorPosition({}, {}, reqid, True)
        return 0

    def reqQryOrder(self, data, reqid):
        if self.has_active:
            return super().reqQryOrder(data, reqid)
        self.onRspQryOrder({}, {}, reqid, True)
        return 0

    def reqOrderInsert(self, data, reqid):
        self.sent.append(dict(data))
        assert data["VolumeTotalOriginal"] == 1
        assert data["CombOffsetFlag"] in {"3", "4"}
        raw = {"BrokerID": "9999", "InvestorID": "demo", "InstrumentID": data["InstrumentID"],
            "ExchangeID": data["ExchangeID"], "OrderRef": data["OrderRef"],
            "FrontID": 1, "SessionID": 2, "OrderSysID": "CLOSE-SYS"}
        self.onRtnOrder({**raw, "OrderStatus": "3"})
        if self.fill:
            self.gross = {}
            self.onRtnTrade({**raw, "TradeID": "CLOSE-TRADE", "Volume": 1, "Price": data["LimitPrice"]})
        return 0

    def reqOrderAction(self, data, reqid):
        super().reqOrderAction(data, reqid)
        self.onRtnOrder({"BrokerID": "9999", "InvestorID": "demo", "InstrumentID": data["InstrumentID"],
            "ExchangeID": data["ExchangeID"], "OrderRef": data["OrderRef"],
            "FrontID": 1, "SessionID": 2, "OrderSysID": "CLOSE-SYS", "OrderStatus": "5"})
        return 0


class CloseExistingTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sequence = 0

    def args(self, *extra, submit=False):
        command = ["--connect", "--symbol", "rb2701", "--exchange", "SHFE",
            "--price-increment", "1", "--multiplier", "10", "--query-timeout", "0.02",
            "--wait-seconds", "0.01"]
        if submit:
            command.extend(["--submit", "--confirm-simnow", "--side", "BUY",
                "--expected-trading-day", "20260922", "--price", "3100"])
        with redirect_stderr(StringIO()):
            return entry.parse_args([*command, *extra])

    def transport(self, **overrides):
        api = type("CaseTdApi", (CloseTdApi,), {**overrides, "created": []})
        self.sequence += 1
        transport = CtpTdApiTransport(client_id=entry.CLIENT_ID, account_id="demo", investor_id="demo",
            front="tcp://fake:1234", broker_id="9999", password="fake", app_id="fake", auth_code="fake",
            td_api_base=api, timeout_seconds=0.02, flow_path=str(self.root / str(self.sequence)))
        return transport, api

    def test_default_precheck_has_no_orders_and_closes_api(self):
        transport, api = self.transport()
        with redirect_stdout(StringIO()):
            result = entry.run_session(self.args(), transport)
        self.assertEqual(result["status"], "prechecked")
        self.assertEqual(result["side"], "BUY")
        self.assertEqual(result["position_effect"], "CLOSE_TODAY")
        self.assertEqual(result["orders_submitted"], 0)
        self.assertEqual(result["final_gross"], {"rb2701.SHFE": (Decimal(0), Decimal(1))})
        self.assertEqual(api.created[0].sent, [])
        self.assertTrue(api.created[0].exited)
        self.assertIsNone(transport._api)

    def test_today_yesterday_and_long_direction_close_exactly_one_without_opening(self):
        for day, amounts, side, offset in (("1", (0, 1), "BUY", "3"),
                ("2", (0, 1), "BUY", "4"), ("1", (1, 0), "SELL", "3")):
            with self.subTest(day=day, side=side):
                transport, api = self.transport(position_day=day, initial_gross={"rb2701.SHFE": amounts})
                with redirect_stdout(StringIO()):
                    result = entry.run_session(self.args("--side", side, submit=True), transport)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["final_gross"], {})
                self.assertEqual(result["final_active_orders"], 0)
                self.assertEqual(result["cleanup_errors"], [])
                self.assertEqual(result["orders_submitted"], 1)
                self.assertEqual(len(api.created[0].sent), 1)
                self.assertEqual(api.created[0].sent[0]["CombOffsetFlag"], offset)
                self.assertEqual(api.created[0].sent[0]["Direction"], "0" if side == "BUY" else "1")
                self.assertTrue(api.created[0].exited)

    def test_empty_multiple_hedged_and_other_contract_positions_block_orders(self):
        for gross in ({}, {"rb2701.SHFE": (0, 2)}, {"rb2701.SHFE": (1, 1)},
                {"rb2701.SHFE": (0, 1), "rb2610.SHFE": (1, 0)}):
            with self.subTest(gross=gross):
                transport, api = self.transport(initial_gross=gross)
                with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "恰好1手"):
                    entry.run_session(self.args(submit=True), transport)
                self.assertEqual(api.created[0].sent, [])
                self.assertTrue(api.created[0].exited)

    def test_unknown_position_date_hedge_flag_and_changed_details_block_orders(self):
        for overrides, reason in (({"position_day": None}, "PositionDate"),
                ({"hedge_flag": None}, "投机仓"), ({"hedge_flag": "3"}, "投机仓"),
                ({"change_on_details": True}, "明细不一致")):
            with self.subTest(overrides=overrides):
                transport, api = self.transport(**overrides)
                with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, reason):
                    entry.run_session(self.args(submit=True), transport)
                self.assertEqual(api.created[0].sent, [])
                self.assertTrue(api.created[0].exited)

    def test_direction_and_td_day_pins_block_orders(self):
        for extra, reason in ((("--side", "SELL"), "方向不一致"),
                (("--expected-trading-day", "20260923"), "交易日变化")):
            transport, api = self.transport()
            with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, reason):
                entry.run_session(self.args(*extra, submit=True), transport)
            self.assertEqual(api.created[0].sent, [])
            self.assertTrue(api.created[0].exited)

    def test_active_orders_block_submission(self):
        transport, api = self.transport(has_active=True)
        with redirect_stdout(StringIO()), self.assertRaises(RuntimeError):
            entry.run_session(self.args(submit=True), transport)
        self.assertEqual(api.created[0].sent, [])
        self.assertTrue(api.created[0].exited)

    def test_unfilled_order_is_canceled_and_not_automatically_resent(self):
        transport, api = self.transport(fill=False)
        with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "完整平仓成交"):
            entry.run_session(self.args(submit=True), transport)
        self.assertEqual(len(api.created[0].sent), 1)
        self.assertIn("cancel", api.created[0].request_names)
        self.assertEqual(api.created[0].gross, {"rb2701.SHFE": (0, 1)})
        self.assertTrue(api.created[0].exited)

    def test_position_details_are_immutable_and_remove_account_identity(self):
        transport, api = self.transport(position_day="2")
        transport.connect(lambda row: None, lambda row: None, lambda reason: None)
        self.addCleanup(transport.close)
        with self.assertRaisesRegex(RuntimeError, "尚未完成登录结算确认或已断线"):
            transport.query_position_details()
        transport.activate()
        rows = transport.query_position_details()
        self.assertIsInstance(rows, tuple)
        self.assertEqual(rows[0]["PositionDate"], "2")
        self.assertEqual(rows[0]["Position"], Decimal(1))
        self.assertNotIn("InvestorID", rows[0])
        with self.assertRaises(TypeError):
            rows[0]["Position"] = 0

    def test_position_details_need_final_fragment_and_correct_account(self):
        for overrides, error in (({"drop_last": True}, TimeoutError), ({"foreign": True}, RuntimeError)):
            transport, api = self.transport(**overrides)
            transport.connect(lambda row: None, lambda row: None, lambda reason: None)
            try:
                transport.activate()
                with self.assertRaises(error):
                    transport.query_position_details()
            finally:
                transport.close()
            self.assertTrue(api.created[0].exited)

    def test_cli_requires_explicit_authorization_day_and_valid_price(self):
        for extra in (("--submit",), ("--confirm-simnow",), ("--price", "NaN"),
                ("--price", "5001"), ("--price", "bad"), ("--wait-seconds", "nan"),
                ("--expected-trading-day", "20260931")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.args(*extra)

    def test_position_details_reject_invalid_lot_quantities(self):
        for quantity in (Decimal("NaN"), Decimal("Infinity"), Decimal("-1"), Decimal("0.5"), "bad"):
            with self.subTest(quantity=quantity):
                transport, api = self.transport(initial_gross={"rb2701.SHFE": (0, quantity)})
                transport.connect(lambda row: None, lambda row: None, lambda reason: None)
                try:
                    transport.activate()
                    with self.assertRaisesRegex(RuntimeError, "明细数量无效"):
                        transport.query_position_details()
                finally:
                    transport.close()

    def test_shared_account_lock_blocks_before_api_is_created(self):
        transport, api = self.transport()
        with account_lock("demo"), self.assertRaisesRegex(RuntimeError, "已有进程"):
            entry.run_session(self.args(submit=True), transport)
        self.assertEqual(api.created, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
