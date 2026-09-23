"""P4-CTP7：无网络验证柜台全量双向今昨仓。"""

from __future__ import annotations

from decimal import Decimal

from market.basic.base import InstrumentId
from strategy.execution.contracts import OrderSide, PositionEffect
from strategy.execution.ctp.ledger import CtpPositionLedger
from strategy.execution.ctp.native_driver import CtpNativeTraderDriver
from strategy.execution.events import AccountPositionEvent, InstrumentPosition
from run_p4_ctp_driver import FakeTraderTransport, _intent
from run_p4_ctp_td_transport import FakeTdApi, _transport


class PositionTdApi(FakeTdApi):
    rows = (
        {"PosiDirection": "2", "Position": 3, "TodayPosition": 1, "YdPosition": 9},
        {"PosiDirection": "2", "Position": 2, "TodayPosition": 2, "YdPosition": 0},
        {"PosiDirection": "3", "Position": 4, "TodayPosition": 1, "YdPosition": 8},
    )

    def reqQryInvestorPosition(self, data, reqid):
        self.request_names.append("positions")
        if not self.rows and not self.drop_position_last:
            self.onRspQryInvestorPosition({}, {}, reqid, True)
            return 0
        for index, detail in enumerate(self.rows):
            self.onRspQryInvestorPosition({
                "BrokerID": "9999", "InvestorID": "demo",
                "InstrumentID": "rb2704", "ExchangeID": "SHFE", **detail,
            }, {}, reqid, index == len(self.rows) - 1 and not self.drop_position_last)
        return 0


def main() -> None:
    transport = _transport(PositionTdApi)
    transport.connect(lambda row: None, lambda row: None, lambda reason: None)
    transport.activate()
    event = transport.query_position_event()
    position = event.positions["rb2704.SHFE"]
    assert (position.net_quantity, position.long_quantity, position.short_quantity) == (
        Decimal(1), Decimal(5), Decimal(4),
    )
    assert (position.long_today, position.long_yesterday,
            position.short_today, position.short_yesterday) == (3, 2, 1, 3)
    assert transport.query_positions() == {"rb2704.SHFE": Decimal(1)}
    assert transport.query_position_event().revision == 2
    print("P4-CTP7a通过：多空、今昨仓分片聚合；静态YdPosition不误作当前昨仓")

    api = transport._api
    api.rows = ({"PosiDirection": "2", "Position": 1},)
    try:
        transport.query_position_event()
    except RuntimeError as error:
        assert "今仓" in str(error)
    else:
        raise AssertionError("缺少TodayPosition不得生成权威快照")
    assert transport._position_revision == 2
    api.rows = ({"PosiDirection": "2", "Position": 1, "TodayPosition": 2},)
    try:
        transport.query_position_event()
    except RuntimeError as error:
        assert "数量" in str(error)
    else:
        raise AssertionError("今仓超过总仓不得生成权威快照")
    assert transport._position_revision == 2
    api.rows = PositionTdApi.rows
    api.drop_position_last = True
    try:
        transport.query_position_event()
    except TimeoutError:
        pass
    else:
        raise AssertionError("缺最后分片不得使用部分仓位")
    assert transport._position_revision == 2
    api.drop_position_last = False
    api.rows = ()
    empty = transport.query_position_event()
    assert empty.revision == 3 and not empty.positions
    transport.close()
    print("P4-CTP7b通过：缺字段、矛盾、超时不推进版本；完整空仓可确认为空")

    transport = _transport(PositionTdApi)
    driver = CtpNativeTraderDriver("ctp-demo", "demo-account", transport)
    driver.start(lambda report: None)
    instrument = InstrumentId.from_str("rb2704.SHFE")
    ledger = CtpPositionLedger("20260922")
    for side, quantity in ((OrderSide.BUY, 3), (OrderSide.SELL, 1)):
        ledger.apply_fill(instrument, side, PositionEffect.OPEN, quantity, 3100, 10)
    try:
        driver.verify_position_ledger(ledger)
    except RuntimeError as error:
        assert "不一致" in str(error)
    else:
        raise AssertionError("数量不一致不得通过账本核对")
    assert ledger.snapshot(instrument).long_today == 3
    ledger.apply_fill(instrument, OrderSide.BUY, PositionEffect.OPEN, 2, 3100, 10)
    ledger.apply_fill(instrument, OrderSide.SELL, PositionEffect.OPEN, 3, 3100, 10)
    # 假柜台含昨仓；先结算转昨，再恢复相同的今仓分布。
    ledger.settle("20260923", {instrument: Decimal(3100)}, {instrument: Decimal(10)})
    ledger.apply_fill(instrument, OrderSide.BUY, PositionEffect.OPEN, 3, 3100, 10)
    ledger.apply_fill(instrument, OrderSide.SELL, PositionEffect.OPEN, 1, 3100, 10)
    # 调整昨仓至柜台的2多/3空，保留已有成本价。
    ledger.apply_fill(instrument, OrderSide.SELL, PositionEffect.CLOSE_YESTERDAY, 3, 3100, 10)
    ledger.apply_fill(instrument, OrderSide.BUY, PositionEffect.CLOSE_YESTERDAY, 1, 3100, 10)
    event = driver.verify_position_ledger(ledger)
    assert event.positions[str(instrument)].net_quantity == 1
    assert ledger.snapshot(instrument).long_today_basis == 3100
    driver.stop()
    print("P4-CTP8通过：Driver权威快照与账本按双向今昨仓核对；不一致不改写成本价")

    class GateTransport(FakeTraderTransport):
        def __init__(self):
            super().__init__()
            self.detail = InstrumentPosition(0, 0, 0, 0, 0, 0, 0)
            self.revision = 0

        def query_position_event(self):
            self.revision += 1
            return AccountPositionEvent(
                "ctp-demo", "demo-account", self.revision, self.revision,
                {"rb2704.SHFE": self.detail},
            )

    gate_transport = GateTransport()
    gate_ledger = CtpPositionLedger("20260922")
    gate_ledger.apply_fill(instrument, OrderSide.BUY, PositionEffect.OPEN, 2, 3100, 10)
    gate_driver = CtpNativeTraderDriver(
        "ctp-demo", "demo-account", gate_transport,
        enable_test_orders=True, disconnect_handler=lambda reason: None,
        position_ledger=gate_ledger,
    )
    gate_driver.start(lambda report: None)
    for action in (lambda: gate_driver.submit_order(_intent()), gate_driver.reconcile):
        try:
            action()
        except RuntimeError:
            pass
        else:
            raise AssertionError("未核对或仓位冲突时不能越过闸门")
    assert not gate_transport.sent
    gate_transport.detail = InstrumentPosition(2, 2, 0, 2, 0, 0, 0)
    assert gate_driver.reconcile()["rb2704.SHFE"] == 2
    gate_driver.submit_order(_intent())
    assert len(gate_transport.sent) == 1
    gate_driver.on_disconnect(4097)
    gate_driver.start(lambda report: None)
    try:
        gate_driver.submit_order(_intent())
    except RuntimeError as error:
        assert "核对" in str(error)
    else:
        raise AssertionError("重连不得沿用旧仓位核对结果")
    gate_driver.reconcile()
    gate_driver.stop()
    print("P4-CTP9通过：启动/重连前禁单，账本不一致闭闸，重新权威核对后仅假柜台可下单")


if __name__ == "__main__":
    main()
