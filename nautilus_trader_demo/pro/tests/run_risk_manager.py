"""F1c：统一下单前风控和Kill Switch分阶段测试。"""

from __future__ import annotations

import argparse
from decimal import Decimal

from market.basic.base import InstrumentId, InstrumentMeta, make_quote_tick
from strategy import (
    BackendExecutionClient,
    ExecutionRequest,
    KillSwitchMode,
    MarketReferencePriceStore,
    NautilusLiveExecutionBackend,
    NetTargetOrderPlanner,
    OrderIntent,
    OrderSide,
    PositionManager,
    PreTradeRiskManager,
    RiskLimits,
    RiskRejected,
    RiskViolationCode,
)


BTC = InstrumentId.from_str("BTCUSDT.BINANCE")


def _intent(quantity: int = 1) -> OrderIntent:
    return OrderIntent(
        strategy_id="f1c-alpha",
        backend_id="live-account",
        instrument_id=BTC,
        side=OrderSide.BUY,
        quantity=quantity,
    )


def test1_limits_and_market_freshness() -> None:
    """F1c1：参考价、批次预计仓位、数量、名义金额和行情时效限制。"""

    positions = PositionManager()
    prices = MarketReferencePriceStore()
    quote = make_quote_tick(
        BTC,
        "99.00",
        "101.00",
        "1.0000",
        "2.0000",
        100,
        meta=InstrumentMeta(BTC, price_precision=2, size_precision=4),
    )
    prices.on_market_event(quote)
    assert prices.get(BTC).price == 100
    prices.update(BTC, 90, 99)  # 旧行情不能覆盖更新行情。
    assert prices.get(BTC).price == 100

    risk = PreTradeRiskManager(
        "live-account",
        positions,
        prices,
        default_limits=RiskLimits(
            max_order_quantity=2,
            max_abs_position=3,
            max_order_notional=250,
            max_abs_position_notional=300,
            max_market_age_ns=10,
        ),
    )
    assert risk.evaluate((_intent(2),), now_ns=105).allowed

    # 两张各2手虽然单笔都合规，但批次预计仓位为4，必须整体拒绝。
    decision = risk.evaluate((_intent(2), _intent(2)), now_ns=105)
    assert not decision.allowed
    codes = {item.code for item in decision.violations}
    assert RiskViolationCode.POSITION_LIMIT in codes
    assert RiskViolationCode.POSITION_NOTIONAL in codes

    stale = risk.evaluate((_intent(1),), now_ns=111)
    assert RiskViolationCode.MARKET_STALE in {
        item.code for item in stale.violations
    }
    oversized = risk.evaluate((_intent(3),), now_ns=105)
    assert RiskViolationCode.ORDER_QUANTITY in {
        item.code for item in oversized.violations
    }
    assert RiskViolationCode.ORDER_NOTIONAL in {
        item.code for item in oversized.violations
    }
    print("F1c1通过：数量、预计仓位、名义金额和行情时效限制正常")


class _RiskDriver:
    driver_id = "live-account"

    def __init__(self, positions=None) -> None:
        self.positions = dict(positions or {})
        self.orders: list[OrderIntent] = []
        self.canceled_strategies: list[str] = []

    def start(self, report_sink) -> None:
        self.report_sink = report_sink

    def stop(self) -> None:
        pass

    def submit_order(self, order: OrderIntent) -> None:
        self.orders.append(order)

    def cancel_strategy(self, strategy_id: str) -> None:
        self.canceled_strategies.append(strategy_id)

    def reconcile(self):
        return dict(self.positions)


def _request(target: int, ts_event: int = 105) -> ExecutionRequest:
    return ExecutionRequest(
        strategy_id="f1c-alpha",
        revision=1,
        client_id="live-account",
        ts_event=ts_event,
        targets={BTC: Decimal(target)},
        execution_policy="DIRECT",
    )


def _client(*, account_position: int = 0):
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    prices.update(BTC, 80_000, 100)
    risk = PreTradeRiskManager(
        "live-account",
        positions,
        prices,
        default_limits=RiskLimits(
            max_order_quantity=2,
            max_abs_position=3,
            max_order_notional=160_000,
            max_abs_position_notional=240_000,
            max_market_age_ns=10,
        ),
    )
    driver = _RiskDriver({BTC: account_position} if account_position else {})
    backend = NautilusLiveExecutionBackend("live-account", driver)
    client = BackendExecutionClient(
        "live-account",
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=risk,
    )
    client.start()
    return client, driver, positions, risk


def test2_client_pretrade_gate() -> None:
    """F1c2：整个Planner输出批次先通过风控，拒绝时不得产生部分副作用。"""

    client, driver, positions, _ = _client()
    client.submit_targets(_request(1))
    assert len(driver.orders) == 1
    assert positions.working_quantity("live-account", BTC) == 1

    try:
        client.submit_targets(_request(4))
    except RiskRejected as error:
        assert RiskViolationCode.ORDER_QUANTITY in {
            item.code for item in error.violations
        }
    else:
        raise AssertionError("超限订单必须被前置风控拒绝")
    assert len(driver.orders) == 1
    assert positions.working_quantity("live-account", BTC) == 1
    client.stop()
    print("F1c2通过：风控位于Planner与Backend之间，拒绝批次没有下单副作用")


def test3_reduce_only_and_kill_switch() -> None:
    """F1c3：REDUCE_ONLY只允许降风险，HALTED撤单并禁止新增订单。"""

    client, driver, _, _ = _client(account_position=2)
    client.set_risk_mode(KillSwitchMode.REDUCE_ONLY, cancel_active_orders=False)
    client.submit_targets(_request(0))
    assert len(driver.orders) == 1
    assert driver.orders[0].side is OrderSide.SELL
    assert driver.orders[0].reduce_only

    try:
        client.submit_targets(_request(3))
    except RiskRejected as error:
        assert RiskViolationCode.REDUCE_ONLY in {
            item.code for item in error.violations
        }
    else:
        raise AssertionError("REDUCE_ONLY状态不能增加风险")

    client.set_risk_mode(KillSwitchMode.HALTED)
    assert driver.canceled_strategies == ["f1c-alpha"]
    try:
        client.submit_targets(_request(1))
    except RiskRejected as error:
        assert RiskViolationCode.KILL_SWITCH in {
            item.code for item in error.violations
        }
    else:
        raise AssertionError("HALTED状态必须拒绝新订单")
    assert len(driver.orders) == 1
    client.stop()
    print("F1c3通过：只减仓模式和Kill Switch撤单/禁单规则正常")


STAGES = {
    1: test1_limits_and_market_freshness,
    2: test2_client_pretrade_gate,
    3: test3_reduce_only_and_kill_switch,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="F1c统一前置风控测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for function in selected.values():
        function()


if __name__ == "__main__":
    main()
