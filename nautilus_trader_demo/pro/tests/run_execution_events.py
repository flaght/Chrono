"""P0：模拟与实盘共用的双向执行事件契约，无网络、无真实下单。"""

from __future__ import annotations

import argparse
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

from trader.execution.events import (
    AccountPositionEvent,
    AccountStateEvent,
    CurrencyBalance,
    ExecutionIdentity,
    FillEvent,
    InstrumentPosition,
    OrderEventStatus,
    OrderUpdateEvent,
)


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity("sim-ctp", "account-a", "ema", "order-1", "rb2704.SHFE")


def _order(status: OrderEventStatus, *, filled: str, event_id: str) -> OrderUpdateEvent:
    return OrderUpdateEvent(
        _identity(), event_id, status, "BUY", "2", filled, 100,
    )


def _invalid(function: object, exception: type[Exception] = ValueError) -> None:
    try:
        function()  # type: ignore[operator]
    except exception:
        return
    raise AssertionError(f"应拒绝非法契约：{function!r}")


def test1_order_and_fill() -> None:
    """订单累计量与成交增量不同；关联键能跨受理、部分成交和全成保留。"""
    submitted = _order(OrderEventStatus.SUBMITTED, filled="0", event_id="report-1")
    accepted = replace(submitted, event_id="report-2", status=OrderEventStatus.ACCEPTED,
                       identity=replace(submitted.identity, venue_order_id="exchange-7"))
    partial = replace(accepted, event_id="report-3", status=OrderEventStatus.PARTIALLY_FILLED,
                      cumulative_filled="1")
    filled = replace(partial, event_id="report-4", status=OrderEventStatus.FILLED,
                     cumulative_filled="2")
    for event in (submitted, accepted, partial, filled):
        assert event.identity.client_order_id == "order-1"
        assert event.identity.strategy_id == "ema"
    assert submitted.identity.venue_order_id is None
    assert accepted.identity.venue_order_id == "exchange-7"
    assert partial.cumulative_filled == Decimal(1)
    assert filled.cumulative_filled == Decimal(2)
    assert len({event.dedupe_key for event in (submitted, accepted, partial, filled)}) == 4
    rejected = OrderUpdateEvent(_identity(), "report-5", "REJECTED", "BUY", 2, 0, 102,
                                reason="柜台拒绝")
    canceled = OrderUpdateEvent(_identity(), "report-6", "CANCELED", "BUY", 2, 1, 103)
    assert rejected.reason == "柜台拒绝"
    assert canceled.cumulative_filled == Decimal(1)  # 部分成交后撤余量。

    trade = FillEvent(accepted.identity, "fill-event-1", "trade-9", "BUY", "1", "3126",
                      "1.5", "CNY", 101, cumulative_filled="1", position_effect="OPEN")
    assert trade.quantity == Decimal(1)
    assert trade.cumulative_filled == Decimal(1)
    assert trade.dedupe_key == ("sim-ctp", "account-a", "order-1", "trade-9")
    assert replace(trade, event_id="resend").dedupe_key == trade.dedupe_key
    unknown_fee = replace(trade, commission=None, commission_currency=None)
    assert unknown_fee.commission is None  # 柜台未报手续费，不代表免费。
    print("P0a通过：订单、成交、原生订单号及去重关联键契约正常")


def test2_account_snapshots() -> None:
    """权威仓位是账户全量快照，CTP今昨仓可表达；资金仍属账户。"""
    position = InstrumentPosition("1", "2", "1", "1", "1", "0", "1")
    positions = AccountPositionEvent("sim-ctp", "account-a", 3, 102,
                                     {"rb2704.SHFE": position})
    assert positions.positions["rb2704.SHFE"].net_quantity == Decimal(1)
    assert positions.positions["rb2704.SHFE"].long_today == Decimal(1)
    empty = AccountPositionEvent("sim-ctp", "account-a", 4, 103, {})
    assert not empty.positions  # 权威确认空仓，不是查询失败。
    balance = CurrencyBalance("CNY", equity="100000", available="80000", margin_used="20000")
    account = AccountStateEvent("sim-ctp", "account-a", 7, 104, {"CNY": balance})
    assert account.balances["CNY"].total is None  # 未提供，不伪造余额。
    _invalid(lambda: positions.positions.__setitem__("rb2704.SHFE", position), AttributeError)
    _invalid(lambda: setattr(account, "revision", 8), FrozenInstanceError)
    print("P0b通过：账户权威仓位、今昨仓、多币种资金及未知值语义正常")


def test3_invalid_fields() -> None:
    """缺失关联ID、矛盾数量、非数值/非法时间不能进入统一事件流。"""
    identity = _identity()
    _invalid(lambda: replace(identity, account_id=" "))
    _invalid(lambda: replace(identity, venue_order_id=" "))
    _invalid(lambda: _order(OrderEventStatus.FILLED, filled="1", event_id="bad"))
    _invalid(lambda: _order(OrderEventStatus.PARTIALLY_FILLED, filled="0", event_id="bad"))
    _invalid(lambda: _order(OrderEventStatus.ACCEPTED, filled="1", event_id="bad"))
    _invalid(lambda: replace(_order(OrderEventStatus.SUBMITTED, filled="0", event_id="ok"),
                             order_quantity="NaN"))
    _invalid(lambda: replace(_order(OrderEventStatus.SUBMITTED, filled="0", event_id="ok"),
                             ts_event=-1))
    _invalid(lambda: replace(_order(OrderEventStatus.SUBMITTED, filled="0", event_id="ok"),
                             sequence=True))
    _invalid(lambda: OrderUpdateEvent(identity, "bad", "REJECTED", "BUY", 1, 0, 100))
    _invalid(lambda: FillEvent(identity, "bad", "", "BUY", 1, 100, None, None, 100))
    _invalid(lambda: FillEvent(identity, "bad", "trade-1", "BUY", 1, 100, None, "CNY", 100))
    _invalid(lambda: FillEvent(identity, "bad", "trade-1", "BUY", 1, 100, -1, "CNY", 100))
    _invalid(lambda: FillEvent(identity, "bad", "trade-1", "BUY", 1, 100, 1, "CNY", 100,
                               cumulative_filled="0"))
    _invalid(lambda: FillEvent(identity, "bad", "trade-1", "BUY", 1, 100, 1, "CNY", 100,
                               position_effect="INVALID"))
    _invalid(lambda: InstrumentPosition("1", "1", "1"))
    _invalid(lambda: InstrumentPosition("1", "1", "0", long_today="2", long_yesterday="0"))
    _invalid(lambda: AccountPositionEvent("sim", "account", 0, 100, {}))
    _invalid(lambda: CurrencyBalance("CNY"))
    _invalid(lambda: AccountStateEvent("sim", "account", 1, 100,
                                       {"USD": CurrencyBalance("CNY", total=1)}))
    print("P0c通过：缺ID、矛盾数量、时间、今昨仓及账户字段均被拒绝")


STAGES = {"order": test1_order_and_fill, "account": test2_account_snapshots,
          "invalid": test3_invalid_fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    for name, function in STAGES.items():
        if args.stage in (name, "all"):
            function()
    print("P0 execution event contracts OK")


if __name__ == "__main__":
    main()
