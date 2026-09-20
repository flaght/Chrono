"""阶段E5：Nautilus通用模拟Backend的分步测试。

E5a验证Profile映射；E5b用内存Engine替身验证Backend所有权、静态装配、
历史/流式行情推进和释放边界。两个阶段都不会发送真实订单。
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal

from bomber.backtest.config import BacktestEngineConfig
from bomber.backtest.models import PerContractFeeModel
from bomber.model import Money, Price, Quantity, Symbol, Venue
from bomber.model.currencies import CNY
from bomber.model.enums import AccountType, AssetClass, BookType, OmsType
from bomber.model.identifiers import InstrumentId, TraderId
from bomber.model.instruments import FuturesContract

from market.basic.base import InstrumentMeta, make_bar
from strategy import (
    ExecutionReport,
    ExecutionReportType,
    GenericVenueProfile,
    NautilusSimExecutionBackend,
    OrderIntent,
    OrderSide,
    SimExecutionBackendPort,
    VenueSimulationProfilePort,
)


def test1_generic_profile_mapping() -> None:
    fee_model = PerContractFeeModel(Money(1, CNY))
    profile = GenericVenueProfile(
        profile_id="generic-shfe",
        venue="SHFE",
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, CNY)],
        base_currency=CNY,
        default_leverage=Decimal("10"),
        fee_model=fee_model,
        book_type=BookType.L1_MBP,
        bar_execution=True,
        trade_execution=True,
        use_reduce_only=True,
    )

    assert isinstance(profile, VenueSimulationProfilePort)
    config = profile.build_backend_config()
    assert profile.venue == Venue("SHFE")
    assert config["venue"] == Venue("SHFE")
    assert config["oms_type"] is OmsType.NETTING
    assert config["account_type"] is AccountType.MARGIN
    assert config["starting_balances"] == [Money(1_000_000, CNY)]
    assert config["default_leverage"] == Decimal("10")
    assert config["fee_model"] is fee_model
    assert config["bar_execution"] is True
    assert config["trade_execution"] is True

    # build_backend_config每次返回独立的容器，外部修改不能污染Profile。
    config["starting_balances"].clear()
    assert len(profile.build_backend_config()["starting_balances"]) == 1
    print("E5a通过：GenericVenueProfile可无损映射Nautilus add_venue通用规则")


class _EngineProbe:
    """E5b专用的BacktestEngine内存替身，不包含撮合逻辑。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.disposed = False
        self.ended = False

    def add_venue(self, **kwargs: object) -> None:
        self.calls.append(("add_venue", kwargs))

    def add_instrument(self, instrument: object) -> None:
        self.calls.append(("add_instrument", instrument))

    def add_strategy(self, strategy: object) -> None:
        self.calls.append(("add_strategy", strategy))

    def add_data(self, data: list[object], **kwargs: object) -> None:
        self.calls.append(("add_data", (list(data), kwargs)))

    def run(self, *, streaming: bool = False) -> None:
        self.calls.append(("run", streaming))

    def clear_data(self) -> None:
        self.calls.append(("clear_data", None))

    def end(self) -> None:
        self.ended = True
        self.calls.append(("end", None))

    def get_result(self) -> dict[str, int]:
        return {"calls": len(self.calls)}

    def dispose(self) -> None:
        self.disposed = True
        self.calls.append(("dispose", None))


def _profile() -> GenericVenueProfile:
    return GenericVenueProfile(
        profile_id="generic-shfe",
        venue="SHFE",
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, CNY)],
        base_currency=CNY,
    )


def test2_backend_ownership_and_lifecycle() -> None:
    engine = _EngineProbe()
    backend = NautilusSimExecutionBackend("nautilus-sim", engine=engine)
    assert isinstance(backend, SimExecutionBackendPort)
    assert backend.engine is engine

    profile = _profile()
    instrument = object()
    strategy = object()
    historical_event = object()
    backend.add_profile(profile)
    backend.add_instrument(instrument)
    backend.add_strategy(strategy)
    backend.add_data([historical_event], sort=True)
    assert backend.profiles == (profile,)
    assert [name for name, _ in engine.calls] == [
        "add_venue",
        "add_instrument",
        "add_strategy",
        "add_data",
    ]

    result = backend.run()
    # run前Backend会自动注册唯一内部订单网关，因此比静态装配多一次add_strategy。
    assert result["calls"] == 6
    assert backend.result() == result
    try:
        backend.add_data([object()])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Backend启动后必须禁止修改静态配置")
    backend.stop()
    backend.stop()
    assert engine.disposed is True

    stream_engine = _EngineProbe()
    stream_backend = NautilusSimExecutionBackend("paper-sim", engine=stream_engine)
    stream_backend.add_profile(_profile())
    reports = stream_backend.process_market_event(object())
    assert reports == ()
    assert [name for name, _ in stream_engine.calls][-3:] == [
        "add_data",
        "run",
        "clear_data",
    ]
    stream_backend.stop()
    assert stream_engine.ended is True
    assert stream_engine.disposed is True
    assert stream_backend.result() is not None
    print("E5b通过：NautilusSimExecutionBackend统一管理引擎装配、推进和释放")


E5C_ID = InstrumentId.from_str("rb9998.SHFE")


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000_000)


def _future() -> FuturesContract:
    return FuturesContract(
        instrument_id=E5C_ID,
        raw_symbol=Symbol(E5C_ID.symbol.value),
        asset_class=AssetClass.COMMODITY,
        currency=CNY,
        price_precision=0,
        price_increment=Price.from_str("1"),
        multiplier=Quantity.from_int(10),
        lot_size=Quantity.from_int(1),
        underlying="rb",
        activation_ns=_ns("2025-01-01"),
        expiration_ns=_ns("2030-01-01"),
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.08"),
        exchange="SHFE",
        ts_event=0,
        ts_init=0,
    )


def test3_order_intent_to_native_fill_report() -> None:
    """E5c：真实BacktestEngine将中立订单意图撮合成统一成交回报。"""

    meta = InstrumentMeta(
        instrument_id=E5C_ID,
        price_precision=0,
        size_precision=0,
        price_increment=Decimal(1),
        multiplier=Decimal(10),
        currency="CNY",
        exchange="SHFE",
    )
    prices = [3100, 3101, 3102]
    bars = [
        make_bar(
            instrument_id=E5C_ID,
            open=price,
            high=price + 2,
            low=price - 2,
            close=price + 1,
            volume=100,
            ts_event=1_800_000_000_000_000_000 + index * 60_000_000_000,
            meta=meta,
            bar_type="1-MINUTE",
        )
        for index, price in enumerate(prices)
    ]
    backend = NautilusSimExecutionBackend(
        "e5c-nautilus-sim",
        BacktestEngineConfig(trader_id=TraderId("E5C-TESTER"), run_analysis=False),
    )
    reports: list[ExecutionReport] = []
    backend.register_report_handler(reports.append)
    backend.add_profile(
        GenericVenueProfile(
            profile_id="e5c-shfe",
            venue="SHFE",
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(1_000_000, CNY)],
            base_currency=CNY,
            use_market_order_acks=True,
        ),
    )
    backend.add_instrument(_future())
    backend.add_data(bars)
    backend.submit_order(
        OrderIntent(
            strategy_id="e5c-alpha",
            backend_id=backend.backend_id,
            instrument_id=E5C_ID,
            side=OrderSide.BUY,
            quantity=1,
            metadata={"signal": "test-entry"},
        ),
    )
    try:
        result = backend.run()
        native_orders = backend.engine.trader.generate_orders_report()
        native_fills = backend.engine.trader.generate_fills_report()
        fills = [item for item in reports if item.report_type is ExecutionReportType.FILLED]
        assert not native_orders.empty
        assert not native_fills.empty
        assert len(fills) == 1
        assert fills[0].instrument_id == E5C_ID
        assert fills[0].filled_quantity == Decimal(1)
        assert fills[0].fill_price is not None
        assert fills[0].metadata["strategy_id"] == "e5c-alpha"
        assert fills[0].metadata["signal"] == "test-entry"
        assert result.total_orders == len(native_orders)
        print(
            "E5c通过：OrderIntent已转为Nautilus原生订单并产生"
            f"ExecutionReport，orders={len(native_orders)} fills={len(native_fills)}",
        )
    finally:
        backend.stop()


STAGES = {
    1: test1_generic_profile_mapping,
    2: test2_backend_ownership_and_lifecycle,
    3: test3_order_intent_to_native_fill_report,
}


def main() -> None:
    test1_generic_profile_mapping()
    test2_backend_ownership_and_lifecycle()
    test3_order_intent_to_native_fill_report()

if __name__ == "__main__":
    main()
