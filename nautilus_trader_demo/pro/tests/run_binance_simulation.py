"""阶段E6：Binance USDT永续Profile与离线Bar模拟撮合测试。

E6a只验证Profile和合约工厂；E6b读取market/replay产生的Binance标准Bar，
使用NautilusSimExecutionBackend完成开仓、reduce-only平仓、手续费和归零仓位验证。
E6c验证Profile配置的初始/维持保证金率在Nautilus账户中的实际运行金额。
不会连接Binance网络，也不会发送真实订单。
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.backtest.models import StandardMarginModel
from bomber.model.currencies import BTC, USDT
from bomber.model.data import Bar
from bomber.model.enums import AccountType, OmsType, PositionSide
from bomber.model.identifiers import TraderId

from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import BinanceKlineParser, BinanceMarketType
from trader import (
    BinanceUsdtFuturesProfile,
    ExecutionReportType,
    NautilusSimExecutionBackend,
    OrderIntent,
    OrderSide,
    VenueSimulationProfilePort,
)


BAR_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/binance_data/"
    "futures/um/klines/1m/BTCUSDT/2023-09-02.csv",
)


def _profile() -> BinanceUsdtFuturesProfile:
    # 这些费率是本测试输入，不代表Binance当前或任意历史时点的官方账户费率。
    return BinanceUsdtFuturesProfile(
        starting_balance=Decimal("100000"),
        leverage=Decimal("10"),
        maintenance_margin_rate=Decimal("0.005"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
    )


def _instrument(profile: BinanceUsdtFuturesProfile):
    return profile.make_instrument(
        "BTCUSDT",
        price_precision=2,
        size_precision=6,
        price_increment=Decimal("0.01"),
        size_increment=Decimal("0.000001"),
        base_currency=BTC,
        min_quantity=Decimal("0.000001"),
        min_notional=Decimal("5"),
    )


def test1_binance_profile_and_instrument() -> None:
    profile = _profile()
    config = profile.build_backend_config()
    instrument = _instrument(profile)

    assert isinstance(profile, VenueSimulationProfilePort)
    assert str(profile.venue) == "BINANCE"
    assert config["oms_type"] is OmsType.NETTING
    assert config["account_type"] is AccountType.MARGIN
    assert config["base_currency"] == USDT
    assert config["default_leverage"] == Decimal("10")
    assert isinstance(config["margin_model"], StandardMarginModel)
    assert config["use_position_ids"] is False
    assert config["use_reduce_only"] is True
    assert str(instrument.id) == "BTCUSDT-PERP.BINANCE"
    assert instrument.is_inverse is False
    assert instrument.quote_currency == USDT
    assert instrument.settlement_currency == USDT
    assert instrument.margin_init == Decimal("0.1")
    assert instrument.margin_maint == Decimal("0.005")
    assert instrument.maker_fee == Decimal("0.0002")
    assert instrument.taker_fee == Decimal("0.0005")
    print("E6a通过：Binance USDT永续Profile和线性合约规则映射正常")


def _load_bars(instrument_id) -> list[Bar]:
    feed = FileReplayFeed("E6_BINANCE_BAR_LOADER")
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=2,
            size_precision=6,
            price_increment=Decimal("0.01"),
            multiplier=Decimal(1),
            currency="USDT",
            exchange="BINANCE",
        ),
    )
    feed.add_bar_csv(
        BAR_PATH,
        BinanceKlineParser(
            symbol="BTCUSDT",
            market_type=BinanceMarketType.FUTURES,
            interval="1m",
            include_factors=False,
        ),
    )
    feed.subscribe(instrument_id, DataType.BAR, bar_spec="1-MINUTE")
    feed.connect()
    try:
        return [event for event in feed.load_events() if isinstance(event, Bar)]
    finally:
        feed.disconnect()


def _commission_value(report) -> Decimal:
    raw = str(report.metadata["commission"]).replace("_", "")
    return Decimal(raw.split()[0])


def test2_binance_offline_bar_round_trip() -> None:
    profile = _profile()
    instrument = _instrument(profile)
    bars = _load_bars(instrument.id)
    if len(bars) < 2:
        raise AssertionError("Binance离线文件至少需要两根Bar")

    backend = NautilusSimExecutionBackend(
        "e6-binance-sim",
        BacktestEngineConfig(trader_id=TraderId("E6-TESTER"), run_analysis=False),
    )
    backend.add_profile(profile)
    backend.add_instrument(instrument)
    try:
        backend.submit_order(
            OrderIntent(
                strategy_id="e6-alpha",
                backend_id=backend.backend_id,
                instrument_id=instrument.id,
                side=OrderSide.BUY,
                quantity=Decimal("0.001"),
                metadata={"action": "open"},
            ),
        )
        open_reports = backend.process_market_event(bars[0])
        open_fills = [
            report
            for report in open_reports
            if report.report_type is ExecutionReportType.FILLED
        ]
        assert len(open_fills) == 1
        assert _commission_value(open_fills[0]) > 0

        backend.submit_order(
            OrderIntent(
                strategy_id="e6-alpha",
                backend_id=backend.backend_id,
                instrument_id=instrument.id,
                side=OrderSide.SELL,
                quantity=Decimal("0.001"),
                reduce_only=True,
                metadata={"action": "close"},
            ),
        )
        close_reports = backend.process_market_event(bars[1])
        close_fills = [
            report
            for report in close_reports
            if report.report_type is ExecutionReportType.FILLED
        ]
        assert len(close_fills) == 1
        assert _commission_value(close_fills[0]) > 0

        native_orders = backend.engine.cache.orders()
        assert len(native_orders) == 2
        assert sum(order.is_reduce_only for order in native_orders) == 1
        assert Decimal(str(backend.engine.portfolio.net_position(instrument.id))) == 0
    finally:
        backend.stop()

    result = backend.result()
    assert result is not None
    assert result.total_orders == 2
    print(
        "E6b通过：Binance离线Bar完成开仓、reduce-only平仓和MakerTaker手续费计算，"
        f"bars={len(bars):,} orders={result.total_orders}",
    )


def test3_binance_margin_runtime_semantics() -> None:
    """验证保证金率进入真实MarginAccount后没有被杠杆重复折算。"""

    profile = _profile()
    instrument = _instrument(profile)
    bars = _load_bars(instrument.id)
    if len(bars) < 3:
        raise AssertionError("Binance离线文件至少需要三根Bar")

    backend = NautilusSimExecutionBackend(
        "e6-binance-margin",
        BacktestEngineConfig(trader_id=TraderId("E6-MARGIN"), run_analysis=False),
    )
    backend.add_profile(profile)
    backend.add_instrument(instrument)
    try:
        # 第一根Bar只用于启动引擎并创建真实MarginAccount。
        backend.process_market_event(bars[0])
        account = backend.engine.portfolio.account(profile.venue)
        if account is None:
            raise AssertionError("未创建Binance模拟保证金账户")

        probe_quantity = instrument.make_qty(Decimal("0.001"))
        probe_price = instrument.make_price(Decimal("25801.30"))
        initial_margin = account.calculate_margin_init(
            instrument=instrument,
            quantity=probe_quantity,
            price=probe_price,
            use_quote_for_inverse=False,
        )
        maintenance_margin = account.calculate_margin_maint(
            instrument=instrument,
            side=PositionSide.LONG,
            quantity=probe_quantity,
            price=probe_price,
            use_quote_for_inverse=False,
        )
        assert initial_margin.as_decimal() == Decimal("2.58013000")
        assert maintenance_margin.as_decimal() == Decimal("0.12900650")

        # 再走一次真实订单和持仓更新，确认Portfolio中锁定的是0.5%维持保证金。
        backend.submit_order(
            OrderIntent(
                strategy_id="e6-margin-alpha",
                backend_id=backend.backend_id,
                instrument_id=instrument.id,
                side=OrderSide.BUY,
                quantity=Decimal("0.001"),
                metadata={"action": "margin-open"},
            ),
        )
        open_reports = backend.process_market_event(bars[1])
        open_fill = next(
            report
            for report in open_reports
            if report.report_type is ExecutionReportType.FILLED
        )
        if open_fill.fill_price is None:
            raise AssertionError("开仓成交回报缺少成交价")
        expected_maintenance = (
            open_fill.fill_price
            * open_fill.filled_quantity
            * profile.maintenance_margin_rate
        )
        locked_maintenance = account.margin_maint(instrument.id)
        if locked_maintenance is None:
            raise AssertionError("开仓后没有生成维持保证金")
        assert locked_maintenance.as_decimal() == expected_maintenance
        assert account.balance_locked(USDT).as_decimal() == expected_maintenance

        backend.submit_order(
            OrderIntent(
                strategy_id="e6-margin-alpha",
                backend_id=backend.backend_id,
                instrument_id=instrument.id,
                side=OrderSide.SELL,
                quantity=Decimal("0.001"),
                reduce_only=True,
                metadata={"action": "margin-close"},
            ),
        )
        close_reports = backend.process_market_event(bars[2])
        assert any(
            report.report_type is ExecutionReportType.FILLED
            for report in close_reports
        )
        assert account.margin_maint(instrument.id) is None
        assert account.balance_locked(USDT).as_decimal() == 0
        assert Decimal(str(backend.engine.portfolio.net_position(instrument.id))) == 0
    finally:
        backend.stop()

    print(
        "E6c通过：Binance初始保证金率10%、维持保证金率0.5%运行金额正确，"
        "平仓后保证金释放且仓位归零",
    )


STAGES = {
    1: test1_binance_profile_and_instrument,
    2: test2_binance_offline_bar_round_trip,
    3: test3_binance_margin_runtime_semantics,
}


def main() -> None:
    test1_binance_profile_and_instrument()
    test2_binance_offline_bar_round_trip()
    test3_binance_margin_runtime_semantics()


if __name__ == "__main__":
    main()
