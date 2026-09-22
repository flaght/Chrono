"""P4-CTP3：原生TdApi请求/回调聚合的无网络假API测试。"""

from __future__ import annotations

from decimal import Decimal

from strategy.execution.ctp.td_transport import CtpTdApiTransport


class FakeTdApi:
    def __init__(self):
        self.request_names = []
        self.drop_position_last = False
        self.account_error = False
        self.exited = False

    def createFtdcTraderApi(self, flow_path):
        self.request_names.append("create")

    def subscribePrivateTopic(self, mode):
        assert mode == 2

    def subscribePublicTopic(self, mode):
        assert mode == 2

    def registerFront(self, front):
        assert front == "tcp://fake:1234"

    def init(self):
        self.onFrontConnected()

    def exit(self):
        self.exited = True

    def reqAuthenticate(self, data, reqid):
        self.request_names.append("auth")
        self.onRspAuthenticate({}, {}, reqid, True)
        return 0

    def reqUserLogin(self, data, reqid):
        self.request_names.append("login")
        self.onRspUserLogin(
            {"BrokerID": "9999", "UserID": "demo",
             "TradingDay": "20260922", "MaxOrderRef": "17"}, {}, reqid, True,
        )
        return 0

    def reqSettlementInfoConfirm(self, data, reqid):
        self.request_names.append("settlement")
        self.onRspSettlementInfoConfirm(
            {"BrokerID": "9999", "InvestorID": "demo"}, {}, reqid, True,
        )
        return 0

    def reqQryInvestorPosition(self, data, reqid):
        self.request_names.append("positions")
        base = {"BrokerID": "9999", "InvestorID": "demo",
                "InstrumentID": "rb2704", "ExchangeID": "SHFE"}
        self.onRspQryInvestorPosition(
            {**base, "PosiDirection": "2", "Position": 2}, {}, reqid, False,
        )
        if not self.drop_position_last:
            self.onRspQryInvestorPosition(
                {**base, "PosiDirection": "3", "Position": 1}, {}, reqid, True,
            )
        return 0

    def reqQryTradingAccount(self, data, reqid):
        self.request_names.append("account")
        if self.account_error:
            self.onRspQryTradingAccount({}, {"ErrorID": 42}, reqid, True)
        else:
            self.onRspQryTradingAccount({
                "BrokerID": "9999", "AccountID": "demo", "CurrencyID": "CNY",
                "Balance": 100000, "Available": 80000, "CurrMargin": 20000,
            }, {}, reqid, True)
        return 0

    def reqQryOrder(self, data, reqid):
        self.request_names.append("orders")
        base = {"BrokerID": "9999", "InvestorID": "demo",
                "InstrumentID": "rb2704", "ExchangeID": "SHFE",
                "Direction": "0", "VolumeTotalOriginal": 2}
        self.onRspQryOrder({
            **base, "OrderRef": "18", "OrderStatus": "1",
            "VolumeTraded": 1, "VolumeTotal": 1,
        }, {}, reqid, False)
        self.onRspQryOrder({
            **base, "OrderRef": "16", "OrderStatus": "0",
            "VolumeTraded": 2, "VolumeTotal": 0,
        }, {}, reqid, True)
        return 0

    def reqOrderInsert(self, data, reqid):
        self.request_names.append("insert")
        return 0

    def reqOrderAction(self, data, reqid):
        self.request_names.append("cancel")
        return 0


def _transport(api_base=FakeTdApi):
    return CtpTdApiTransport(
        client_id="ctp-demo", account_id="demo-account",
        front="tcp://fake:1234", broker_id="9999", investor_id="demo",
        password="fake-password", app_id="fake-app", auth_code="fake-auth",
        td_api_base=api_base, timeout_seconds=0.05,
    )


def main() -> None:
    transport = _transport()
    orders, trades, disconnects = [], [], []
    session = transport.connect(orders.append, trades.append, disconnects.append)
    api = transport._api
    assert session.trading_day == "20260922" and session.max_order_ref == 17
    transport.activate()
    assert api.request_names[:4] == ["create", "auth", "login", "settlement"]
    assert transport.query_positions() == {"rb2704.SHFE": Decimal(1)}
    state = transport.query_account()
    assert state.revision == 1 and state.balances["CNY"].available == 80000
    snapshot = transport.query_active_orders()
    assert snapshot.revision == 1 and len(snapshot.orders) == 1
    assert snapshot.orders[0].remaining_quantity == 1
    assert snapshot.orders[0].client_order_id == "CTP-9999-demo-20260922-18"
    print("P4-CTP3a通过：认证登录、结算确认及三种查询均等待本次最终回报")

    api.drop_position_last = True
    try:
        transport.query_positions()
    except TimeoutError:
        pass
    else:
        raise AssertionError("分片查询缺少最终回报不能使用部分仓位")
    api.account_error = True
    try:
        transport.query_account()
    except RuntimeError as error:
        assert "42" in str(error)
    else:
        raise AssertionError("柜台资金错误不能返回旧资金快照")
    assert transport._account_revision == 1
    print("P4-CTP3b通过：缺最后分片和柜台错误均失败闭闸，不推进资金版本")

    api.onRtnOrder({"OrderRef": "18", "OrderStatus": "3"})
    api.onRtnTrade({"OrderRef": "18", "TradeID": "T-1"})
    assert len(orders) == len(trades) == 1
    api.onFrontDisconnected(4097)
    assert disconnects == [4097]
    try:
        transport.query_active_orders()
    except RuntimeError:
        pass
    else:
        raise AssertionError("断线后不能返回旧订单快照")
    transport.close()
    assert api.exited
    print("P4-CTP3c通过：原生推送转交Driver且断线后所有查询闭闸")

    class EarlyPushTdApi(FakeTdApi):
        def init(self):
            self.onRtnOrder({"OrderRef": "unmapped"})
            self.onFrontConnected()

    transport = _transport(EarlyPushTdApi)
    try:
        transport.connect(lambda row: None, lambda row: None, lambda reason: None)
    except RuntimeError as error:
        assert "未归属订单" in str(error)
    else:
        raise AssertionError("Driver尚未激活时的订单推送不能被静默丢弃")
    assert transport._api is None
    print("P4-CTP3d通过：连接期间早到未归属订单使会话失败")

    class RejectTdApi(FakeTdApi):
        def reqOrderInsert(self, data, reqid):
            self.onRspOrderInsert({"OrderRef": data["OrderRef"]},
                                  {"ErrorID": 31, "ErrorMsg": "拒单"}, reqid, True)
            self.onErrRtnOrderInsert({"OrderRef": data["OrderRef"]},
                                     {"ErrorID": 31, "ErrorMsg": "拒单"})
            return 0

    transport = _transport(RejectTdApi)
    orders, disconnects = [], []
    transport.connect(orders.append, lambda row: None, disconnects.append)
    transport.activate()
    transport.send_order({
        "BrokerID": "9999", "InvestorID": "demo", "InstrumentID": "rb2704",
        "ExchangeID": "SHFE", "OrderRef": "18",
    })
    assert len(orders) == 1 and all(row["OrderSubmitStatus"] == "4" for row in orders)
    assert all("31" in row["StatusMsg"] for row in orders)
    assert not disconnects
    transport.close()
    print("P4-CTP4a通过：异步报单拒绝补齐身份并上送Driver，双回调可去重")

    class CancelRejectTdApi(FakeTdApi):
        def reqOrderAction(self, data, reqid):
            self.onRspOrderAction({"OrderRef": data["OrderRef"]},
                                  {"ErrorID": 32, "ErrorMsg": "撤单拒绝"}, reqid, True)
            return 0

    transport = _transport(CancelRejectTdApi)
    disconnects = []
    transport.connect(lambda row: None, lambda row: None, disconnects.append)
    transport.activate()
    transport._api.onRtnOrder({
        "OrderRef": "18", "FrontID": 1, "SessionID": 2,
        "InstrumentID": "rb2704", "ExchangeID": "SHFE",
    })
    transport.cancel_order("18")
    assert disconnects == [-32]
    try:
        transport.query_active_orders()
    except RuntimeError:
        pass
    else:
        raise AssertionError("撤单拒绝后不得继续使用旧活动订单状态")
    transport.close()
    print("P4-CTP4b通过：撤单拒绝不伪造已撤单，关闭查询/下单闸门")


if __name__ == "__main__":
    main()
