"""阶段E10：CTP今昨仓、平仓规划、手续费与逐日结算测试。

测试全部在内存运行，不连接CTP行情或交易柜台。
"""

from __future__ import annotations

import argparse
from decimal import Decimal

from bomber.model.enums import OmsType
from bomber.model.identifiers import InstrumentId

from trader import (
    CtpClosePlanner,
    CtpCommissionRule,
    CtpExecutionAccounting,
    CtpFuturesHedgingProfile,
    CtpPositionLedger,
    ExecutionReport,
    ExecutionReportType,
    ExecutionRequest,
    OrderSide,
    PositionEffect,
)


RB_ID = InstrumentId.from_str("rb2704.SHFE")


def test1_position_ledger_and_daily_settlement() -> None:
    """E10a：双向开仓、逐日盯市、今转昨和指定平今/平昨。"""

    ledger = CtpPositionLedger("20260918")
    ledger.apply_fill(RB_ID, OrderSide.BUY, PositionEffect.OPEN, 5, 3100, 10)
    ledger.apply_fill(RB_ID, OrderSide.SELL, PositionEffect.OPEN, 3, 3110, 10)
    before = ledger.snapshot(RB_ID)
    assert before.long_today == 5 and before.short_today == 3
    assert before.net_position == 2

    settlement = ledger.settle(
        "20260919",
        {RB_ID: Decimal(3120)},
        {RB_ID: Decimal(10)},
    )
    # 多头: (3120-3100)*5*10=1000；空头: (3110-3120)*3*10=-300。
    assert settlement.variation_margin == Decimal(700)
    rolled = ledger.snapshot(RB_ID)
    assert rolled.long_today == 0 and rolled.short_today == 0
    assert rolled.long_yesterday == 5 and rolled.short_yesterday == 3
    assert rolled.long_yesterday_basis == 3120

    ledger.apply_fill(RB_ID, OrderSide.BUY, PositionEffect.OPEN, 2, 3130, 10)
    close_today = ledger.apply_fill(
        RB_ID,
        OrderSide.SELL,
        PositionEffect.CLOSE_TODAY,
        1,
        3140,
        10,
    )
    close_yesterday = ledger.apply_fill(
        RB_ID,
        OrderSide.SELL,
        PositionEffect.CLOSE_YESTERDAY,
        2,
        3140,
        10,
    )
    assert close_today.realized_pnl == Decimal(100)
    assert close_yesterday.realized_pnl == Decimal(400)
    after = ledger.snapshot(RB_ID)
    assert after.long_today == 1 and after.long_yesterday == 3
    assert after.short_yesterday == 3
    try:
        ledger.apply_fill(
            RB_ID,
            OrderSide.SELL,
            PositionEffect.CLOSE_TODAY,
            2,
            3140,
            10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("超过今仓数量的平今必须拒绝")
    print("E10a通过：CTP双向今昨仓、逐日盯市和指定平仓账本正常")


def test2_close_planner_and_commission() -> None:
    """E10b：跨零目标拆成平今、平昨、反向开仓，并区分手续费。"""

    ledger = CtpPositionLedger("20260918")
    ledger.apply_fill(RB_ID, OrderSide.BUY, PositionEffect.OPEN, 3, 3100, 10)
    ledger.settle("20260919", {RB_ID: 3100}, {RB_ID: 10})
    ledger.apply_fill(RB_ID, OrderSide.BUY, PositionEffect.OPEN, 1, 3110, 10)
    planner = CtpClosePlanner(ledger, close_today_first=True)
    request = ExecutionRequest(
        strategy_id="e10-alpha",
        revision=1,
        client_id="ctp-live",
        ts_event=1,
        targets={RB_ID: Decimal(-2)},
        execution_policy="DIRECT",
    )
    orders = planner.plan(request)
    assert len(orders) == 3
    assert [order.position_effect for order in orders] == [
        PositionEffect.CLOSE_TODAY,
        PositionEffect.CLOSE_YESTERDAY,
        PositionEffect.OPEN,
    ]
    assert [order.quantity for order in orders] == [1, 3, 2]
    assert all(order.side is OrderSide.SELL for order in orders)
    assert [order.reduce_only for order in orders] == [True, True, False]

    commission = CtpCommissionRule(
        open_per_contract=1,
        close_per_contract=2,
        close_today_per_contract=5,
    )
    assert commission.calculate(PositionEffect.OPEN, 2) == 2
    assert commission.calculate(PositionEffect.CLOSE_YESTERDAY, 3) == 6
    assert commission.calculate(PositionEffect.CLOSE_TODAY, 1) == 5
    print("E10b通过：CTP目标已生成平今、平昨、反向开仓及差异化手续费")


def _fill(
    order_id: str,
    side: OrderSide,
    effect: PositionEffect,
    quantity: int,
    price: int,
) -> ExecutionReport:
    return ExecutionReport(
        backend_id="ctp-live",
        client_order_id=order_id,
        instrument_id=RB_ID,
        report_type=ExecutionReportType.FILLED,
        ts_event=1,
        filled_quantity=quantity,
        fill_price=price,
        order_side=side,
        order_quantity=quantity,
        position_effect=effect,
    )


def test3_execution_accounting_and_hedging_profile() -> None:
    """E10c：统一成交回报写入CTP账本，HEDGING Profile边界明确。"""

    profile = CtpFuturesHedgingProfile(
        commission_rule=CtpCommissionRule(1, 2, 5),
    )
    config = profile.build_backend_config()
    assert config["oms_type"] is OmsType.HEDGING
    assert config["use_position_ids"] is True
    assert config["use_reduce_only"] is True

    ledger = CtpPositionLedger("20260918")
    accounting = CtpExecutionAccounting(
        ledger,
        {RB_ID: 10},
        {RB_ID: profile.commission_rule},
    )
    accounting.on_report(_fill("O-1", OrderSide.BUY, PositionEffect.OPEN, 2, 3100))
    accounting.on_report(
        _fill("O-2", OrderSide.SELL, PositionEffect.CLOSE_TODAY, 1, 3110),
    )
    snapshot = ledger.snapshot(RB_ID)
    assert snapshot.long_today == 1
    assert accounting.results[-1].realized_pnl == Decimal(100)
    assert accounting.total_commission == Decimal(7)
    settlement = accounting.settle("20260919", {RB_ID: 3120})
    assert settlement.variation_margin == Decimal(200)
    assert ledger.snapshot(RB_ID).long_yesterday == 1
    assert accounting.net_cash_change == Decimal(293)
    print("E10c通过：ExecutionReport已同步CTP账本，HEDGING Profile与每日结算正常")


STAGES = {
    1: test1_position_ledger_and_daily_settlement,
    2: test2_close_planner_and_commission,
    3: test3_execution_accounting_and_hedging_profile,
}


def main() -> None:
    test1_position_ledger_and_daily_settlement()
    test2_close_planner_and_commission()
    test3_execution_accounting_and_hedging_profile()


if __name__ == "__main__":
    main()
