"""Offline check that SimNow EMA reversals close before reopening."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market.basic.base import InstrumentId, InstrumentMeta, make_bar
from trader.contracts import ExecutionRequest
from trader.execution.contracts import OrderSide, PositionEffect
from trader.execution.ctp import CtpPositionLedger
from trader.execution.risk import MarketReferencePriceStore
from examples.single_ema.ema_ctp_simnow import CtpLimitPlanner
from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy


INSTRUMENT = InstrumentId.from_str("rb2701.SHFE")


class Positions:
    working = Decimal(0)

    def working_quantity(self, client_id, instrument_id):
        assert client_id == "ctp-simnow" and instrument_id == INSTRUMENT
        return self.working


class Context:
    def __init__(self):
        self.targets = []

    def submit(self, target):
        self.targets.append(target)


def main() -> None:
    ledger = CtpPositionLedger()
    positions = Positions()
    prices = MarketReferencePriceStore()
    prices.update(INSTRUMENT, Decimal(3100), 1)
    planner = CtpLimitPlanner(ledger, positions, prices, Decimal(1), 1)
    request = ExecutionRequest(
        strategy_id="ema-ctp-simnow", revision=1, client_id="ctp-simnow",
        ts_event=1, targets={INSTRUMENT: Decimal(-1)},
        execution_policy="DIRECT",
    )
    ledger.apply_fill(INSTRUMENT, OrderSide.BUY, PositionEffect.OPEN, 1, 3100, 10)
    first = planner.plan(request)
    assert len(first) == 1
    assert first[0].side is OrderSide.SELL
    assert first[0].position_effect is PositionEffect.CLOSE_TODAY
    positions.working = Decimal(-1)
    assert planner.plan(request) == ()
    positions.working = Decimal(0)
    ledger.apply_fill(INSTRUMENT, OrderSide.SELL, PositionEffect.CLOSE_TODAY, 1, 3100, 10)
    second = planner.plan(request)
    assert len(second) == 1
    assert second[0].side is OrderSide.SELL
    assert second[0].position_effect is PositionEffect.OPEN

    context = Context()
    strategy = EmaCrossTargetStrategy(
        "ema-staging", EmaCrossConfig(
            fast_period=2, slow_period=3,
            repeat_target_each_bar=True, skip_single_price=False,
        ),
    )
    strategy._bind(context)
    meta = InstrumentMeta(
        instrument_id=INSTRUMENT, price_precision=0, size_precision=0,
        price_increment=Decimal(1), multiplier=Decimal(10),
        currency="CNY", exchange="SHFE",
    )
    for index in range(4):
        bar = make_bar(
            instrument_id=INSTRUMENT, open=3100, high=3101, low=3099,
            close=3100, volume=1, ts_event=index + 1,
            meta=meta, bar_type="1-MINUTE",
        )
        strategy.on_bar("primary_bar", bar)
    assert len(context.targets) == 2
    assert context.targets[0].targets == context.targets[1].targets
    print("CTP EMA分阶段反手及下一Bar目标重发通过")


if __name__ == "__main__":
    main()
