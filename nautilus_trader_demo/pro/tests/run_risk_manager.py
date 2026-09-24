"""F1c：统一下单前风控和Kill Switch分阶段测试。"""

from __future__ import annotations

import argparse
from decimal import Decimal

from market.basic.base import InstrumentId, InstrumentMeta, make_quote_tick
from trader import (
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
    SimulationExecutionClient,
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


class _SimRiskBackend:
    backend_id = "rb-sim"

    def __init__(self) -> None:
        self.orders: list[OrderIntent] = []

    def register_report_handler(self, handler) -> None:
        self.report_handler = handler

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def submit_order(self, order: OrderIntent) -> None:
        self.orders.append(order)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id


def test4_rb_multiplier_and_historical_risk_clock() -> None:
    """RB名义金额按10倍合约乘数计算；Tick聚合信号不误判参考价超前。"""

    rb = InstrumentId.from_str("rb2704.SHFE")
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    prices.update(rb, 3000, 200)
    limits = RiskLimits(
        max_order_quantity=2,
        max_abs_position=2,
        max_order_notional=50_000,
        max_abs_position_notional=60_000,
        max_market_age_ns=20,
        contract_multiplier=10,
    )
    risk = PreTradeRiskManager(
        "rb-sim",
        positions,
        prices,
        instrument_limits={rb: limits},
    )
    backend = _SimRiskBackend()
    client = SimulationExecutionClient(
        "rb-sim",
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=risk,
    )
    client.start()
    try:
        # 信号Bar时间早于触发它结束的行情时间：评估时刻取已处理行情200。
        client.submit_targets(
            ExecutionRequest(
                strategy_id="rb-ema",
                revision=1,
                client_id="rb-sim",
                ts_event=190,
                targets={rb: Decimal(1)},
                execution_policy="DIRECT",
            ),
        )
        assert len(backend.orders) == 1
        # 一手名义金额=3000×10=30000；两手超过50000订单上限。
        oversized = OrderIntent(
            strategy_id="rb-ema",
            backend_id="rb-sim",
            instrument_id=rb,
            side=OrderSide.BUY,
            quantity=2,
        )
        assert RiskViolationCode.ORDER_NOTIONAL in {
            violation.code
            for violation in risk.evaluate((oversized,), now_ns=200).violations
        }
        # 没有新行情而信号时间前进，仍须因行情过旧拒绝新增目标。
        try:
            client.submit_targets(
                ExecutionRequest(
                    strategy_id="rb-ema",
                    revision=2,
                    client_id="rb-sim",
                    ts_event=221,
                    targets={rb: Decimal(2)},
                    execution_policy="DIRECT",
                ),
            )
        except RiskRejected as error:
            assert RiskViolationCode.MARKET_STALE in {
                violation.code for violation in error.violations
            }
        else:
            raise AssertionError("过期参考价必须拒绝新增RB目标")
        assert len(backend.orders) == 1
    finally:
        client.stop()
    print("F1c4通过：RB乘数名义金额与Tick聚合历史风控时钟正常")


def test5_missing_reference_price_rejects_cleanly() -> None:
    """无参考行情但启用价格风控时，明确拒单，而不是发生max(int)异常。"""

    rb = InstrumentId.from_str("rb2704.SHFE")
    positions = PositionManager()
    backend = _SimRiskBackend()
    client = SimulationExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id,
            positions,
            MarketReferencePriceStore(),
            instrument_limits={rb: RiskLimits(max_market_age_ns=10)},
        ),
    )
    client.start()
    try:
        try:
            client.submit_targets(
                ExecutionRequest(
                    strategy_id="rb-ema",
                    revision=1,
                    client_id=backend.backend_id,
                    ts_event=100,
                    targets={rb: Decimal(1)},
                    execution_policy="DIRECT",
                ),
            )
        except RiskRejected as error:
            assert RiskViolationCode.MARKET_PRICE_MISSING in {
                violation.code for violation in error.violations
            }
        else:
            raise AssertionError("缺少必需参考价时必须明确拒单")
        assert not backend.orders
        assert positions.working_quantity(backend.backend_id, rb) == 0
    finally:
        client.stop()
    print("F1c5通过：缺少参考价时明确风控拒单，不抛时间计算异常")


STAGES = {
    1: test1_limits_and_market_freshness,
    2: test2_client_pretrade_gate,
    3: test3_reduce_only_and_kill_switch,
    4: test4_rb_multiplier_and_historical_risk_clock,
    5: test5_missing_reference_price_rejects_cleanly,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="F1c统一前置风控测试")
    parser.add_argument("--stage", choices=("1", "2", "3", "4", "5", "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for function in selected.values():
        function()


if __name__ == "__main__":
    main()
