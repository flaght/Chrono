"""阶段A：执行Backend、订单意图、回报和Profile协议的纯内存测试。

本测试不读取行情文件、不连接网络、不创建Nautilus引擎，也不会发送真实订单。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence

from market.basic.base import InstrumentId
from trader import (
    AccountPositionSnapshot,
    ExecutionBackendKind,
    ExecutionBackendPort,
    ExecutionReport,
    ExecutionReportType,
    LiveExecutionBackendPort,
    OrderIntent,
    OrderPlannerPort,
    OrderSide,
    OrderType,
    PositionEffect,
    SimExecutionBackendPort,
    VenueSimulationProfilePort,
)
from trader.contracts import ExecutionRequest


BTC_ID = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
RB_ID = InstrumentId.from_str("rb2704.SHFE")


def _request() -> ExecutionRequest:
    return ExecutionRequest(
        strategy_id="alpha",
        revision=1,
        client_id="sim",
        ts_event=100,
        targets={RB_ID: Decimal(2)},
        logical_targets={"position": Decimal(2)},
        execution_policy="DIRECT",
    )


@dataclass
class _PlannerProbe:
    planner_id: str = "generic-netting"

    def plan(self, request: ExecutionRequest) -> Sequence[OrderIntent]:
        quantity = request.targets[RB_ID]
        return (
            OrderIntent(
                strategy_id=request.strategy_id,
                backend_id=request.client_id,
                instrument_id=RB_ID,
                side=OrderSide.BUY,
                quantity=quantity,
            ),
        )


@dataclass
class _ProfileProbe:
    profile_id: str = "ctp-shfe-basic"
    venue: str = "SHFE"

    def build_backend_config(self) -> Mapping[str, Any]:
        return {"oms_type": "HEDGING", "integer_lots": True}


@dataclass
class _SimBackendProbe:
    backend_id: str = "sim"
    kind: ExecutionBackendKind = ExecutionBackendKind.SIMULATION
    orders: list[OrderIntent] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    report_handlers: list[Callable[[ExecutionReport], None]] = field(default_factory=list)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def submit_order(self, order: OrderIntent) -> None:
        self.orders.append(order)

    def register_report_handler(self, handler: Callable[[ExecutionReport], None]) -> None:
        self.report_handlers.append(handler)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def process_market_event(self, event: Any) -> Sequence[ExecutionReport]:
        self.events.append(event)
        return ()

    def result(self) -> Mapping[str, int]:
        return {"orders": len(self.orders), "events": len(self.events)}


@dataclass
class _LiveBackendProbe:
    backend_id: str = "live"
    kind: ExecutionBackendKind = ExecutionBackendKind.LIVE
    reconciliations: int = 0
    orders: list[OrderIntent] = field(default_factory=list)
    report_handlers: list[Callable[[ExecutionReport], None]] = field(default_factory=list)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def submit_order(self, order: OrderIntent) -> None:
        self.orders.append(order)

    def register_report_handler(self, handler: Callable[[ExecutionReport], None]) -> None:
        self.report_handlers.append(handler)

    def cancel_strategy(self, strategy_id: str) -> None:
        del strategy_id

    def reconcile(self) -> AccountPositionSnapshot:
        self.reconciliations += 1
        return AccountPositionSnapshot(self.backend_id, self.reconciliations, 0, {})


def test1_order_contracts() -> None:
    market = OrderIntent(
        strategy_id="ema",
        backend_id="binance-sim",
        instrument_id=BTC_ID,
        side="BUY",
        quantity="0.001",
        reduce_only=True,
    )
    close_today = OrderIntent(
        strategy_id="cta",
        backend_id="ctp-sim",
        instrument_id=RB_ID,
        side=OrderSide.SELL,
        quantity=2,
        order_type=OrderType.LIMIT,
        price=3100,
        position_effect=PositionEffect.CLOSE_TODAY,
    )
    report = ExecutionReport(
        backend_id="ctp-sim",
        client_order_id="O-1",
        instrument_id=RB_ID,
        report_type=ExecutionReportType.FILLED,
        ts_event=101,
        filled_quantity=2,
        fill_price=3100,
    )
    assert market.quantity == Decimal("0.001")
    assert close_today.position_effect is PositionEffect.CLOSE_TODAY
    assert report.filled_quantity == Decimal(2)
    try:
        OrderIntent("bad", "sim", RB_ID, "BUY", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("零数量订单必须被拒绝")
    print("E1通过：中立订单意图可表达BN reduceOnly和CTP平今，回报字段正常")


def test2_backend_capabilities() -> None:
    sim = _SimBackendProbe()
    live = _LiveBackendProbe()
    assert isinstance(sim, ExecutionBackendPort)
    assert isinstance(sim, SimExecutionBackendPort)
    assert not isinstance(sim, LiveExecutionBackendPort)
    assert isinstance(live, ExecutionBackendPort)
    assert isinstance(live, LiveExecutionBackendPort)
    assert not isinstance(live, SimExecutionBackendPort)
    planned = _PlannerProbe().plan(_request())
    sim.submit_order(planned[0])
    sim.process_market_event({"close": 3100})
    live.reconcile()
    assert sim.result() == {"orders": 1, "events": 1}
    assert live.reconciliations == 1
    print("E2通过：模拟Backend和实盘Backend同级，能力协议保持区分")


def test3_planner_and_profile_ports() -> None:
    planner = _PlannerProbe()
    profile = _ProfileProbe()
    assert isinstance(planner, OrderPlannerPort)
    assert isinstance(profile, VenueSimulationProfilePort)
    orders = planner.plan(_request())
    assert len(orders) == 1
    assert orders[0].quantity == Decimal(2)
    assert profile.build_backend_config()["oms_type"] == "HEDGING"
    print("E3通过：OrderPlanner和VenueProfile可独立替换，不进入策略代码")


STAGES = {
    1: test1_order_contracts,
    2: test2_backend_capabilities,
    3: test3_planner_and_profile_ports,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="执行Backend阶段A契约测试")
    parser.add_argument("--stage", choices=(*(str(i) for i in STAGES), "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for number, function in selected.items():
        print(f"\n--- E{number} ---")
        function()
    print("Execution backend contract tests OK")


if __name__ == "__main__":
    main()
