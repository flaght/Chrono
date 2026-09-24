"""P4-CTP2：原生CTP Driver生命周期、权威查询和回报的假柜台测试。"""

from __future__ import annotations

from decimal import Decimal

from market.basic.base import InstrumentId
from trader.execution.contracts import ExecutionReportType, OrderIntent
from trader.execution.ctp.native_driver import CtpNativeTraderDriver, CtpTraderSession
from trader.execution.events import AccountStateEvent, ActiveOrderSnapshot, CurrencyBalance


class FakeTraderTransport:
    is_test_transport = True

    def __init__(self) -> None:
        self.sent = []
        self.canceled = []
        self.closed = False
        self.on_order = None
        self.on_trade = None
        self.on_disconnect = None
        self.fail_positions = False

    def connect(self, on_order, on_trade, on_disconnect):
        self.on_order = on_order
        self.on_trade = on_trade
        self.on_disconnect = on_disconnect
        return CtpTraderSession("9999", "demo", "20260922", 7)

    def close(self):
        self.closed = True

    def activate(self):
        pass

    def query_positions(self):
        if self.fail_positions:
            raise TimeoutError("本次CTP持仓查询未完成")
        return {"rb2704.SHFE": Decimal(1)}

    def query_account(self):
        return AccountStateEvent(
            "ctp-demo", "demo-account", 1, 100,
            {"CNY": CurrencyBalance("CNY", total=100000, available=80000)},
        )

    def query_active_orders(self):
        return ActiveOrderSnapshot("ctp-demo", "demo-account", 1, 100, ())

    def send_order(self, fields):
        self.sent.append(dict(fields))

    def cancel_order(self, order_ref):
        self.canceled.append(order_ref)


def _intent():
    return OrderIntent(
        strategy_id="alpha", backend_id="ctp-demo",
        instrument_id=InstrumentId.from_str("rb2704.SHFE"),
        side="SELL", quantity=2, order_type="LIMIT", price=3125,
        position_effect="CLOSE_TODAY",
    )


def _raw(ref, **fields):
    result = {
        "BrokerID": "9999", "InvestorID": "demo", "InstrumentID": "rb2704",
        "ExchangeID": "SHFE", "OrderRef": ref,
        "FrontID": 1, "SessionID": 2, "OrderSysID": "123",
    }
    result.update(fields)
    return result


def main() -> None:
    transport = FakeTraderTransport()
    reports = []
    disconnected = []
    driver = CtpNativeTraderDriver(
        "ctp-demo", "demo-account", transport,
        disconnect_handler=disconnected.append,
    )
    driver.start(reports.append)
    assert driver.reconcile()["rb2704.SHFE"] == 1
    assert driver.reconcile_account_state().balances["CNY"].available == 80000
    assert driver.reconcile_active_orders().orders == ()
    try:
        driver.submit_order(_intent())
    except RuntimeError as error:
        assert "默认禁单" in str(error)
    else:
        raise AssertionError("只读Driver不能发单")
    driver.stop()
    assert transport.closed and not transport.sent
    print("P4-CTP2a通过：认证会话、权威三查询与默认禁单生命周期正常")

    transport = FakeTraderTransport()
    reports = []
    disconnected = []
    driver = CtpNativeTraderDriver(
        "ctp-demo", "demo-account", transport,
        enable_test_orders=True, disconnect_handler=disconnected.append,
    )
    driver.start(reports.append)
    driver.submit_order(_intent())
    assert transport.sent[0]["OrderRef"] == "8"
    assert transport.sent[0]["CombOffsetFlag"] == "3"
    transport.on_order(_raw("8", OrderStatus="3"))
    transport.on_trade(_raw("8", TradeID="trade-1", Volume=1, Price=3125))
    transport.on_trade(_raw("8", TradeID="trade-1", Volume=1, Price=3125))
    transport.on_trade(_raw("8", TradeID="trade-2", Volume=1, Price=3126))
    transport.on_order(_raw("8", OrderStatus="0"))
    assert [report.report_type for report in reports] == [
        ExecutionReportType.ACCEPTED,
        ExecutionReportType.PARTIALLY_FILLED,
        ExecutionReportType.FILLED,
    ]
    assert len({report.client_order_id for report in reports}) == 1
    assert reports[1].metadata["trade_id"] == "SHFE:trade-1"
    assert not disconnected
    print("P4-CTP2b通过：OrderRef归属、部分/全部成交及重复回报去重正常")

    try:
        transport.on_order(_raw("999", OrderStatus="3"))
    except RuntimeError as error:
        assert "未知OrderRef" in str(error)
    else:
        raise AssertionError("未知订单不能冒充本策略订单")
    assert disconnected
    driver.stop()

    transport = FakeTraderTransport()
    disconnected = []
    driver = CtpNativeTraderDriver(
        "ctp-demo", "demo-account", transport,
        disconnect_handler=disconnected.append,
    )
    driver.start(lambda report: None)

    transport.fail_positions = True
    try:
        driver.reconcile()
    except TimeoutError:
        pass
    else:
        raise AssertionError("CTP查询失败不能退化为缓存仓位")
    transport.on_disconnect(4097)
    assert disconnected and "4097" in disconnected[-1]
    try:
        driver.reconcile_active_orders()
    except RuntimeError:
        pass
    else:
        raise AssertionError("断线后不能继续返回旧活动订单快照")
    driver.stop()
    print("P4-CTP2c通过：查询失败与前置断线保持闭闸，不使用旧缓存")

    transport = FakeTraderTransport()
    reports = []
    driver = CtpNativeTraderDriver(
        "ctp-demo", "demo-account", transport,
        enable_test_orders=True, disconnect_handler=lambda reason: None,
    )
    driver.start(reports.append)
    driver.submit_order(_intent())
    rejection = _raw("8", OrderSubmitStatus="4", StatusMsg="CTP报单拒绝 31")
    transport.on_order(rejection)
    transport.on_order(rejection)
    assert len(reports) == 1
    assert reports[0].report_type is ExecutionReportType.REJECTED
    assert "31" in reports[0].reason
    assert not driver._orders
    driver.stop()
    print("P4-CTP4c通过：Driver重复拒单回报只生成一次终态")


if __name__ == "__main__":
    main()
