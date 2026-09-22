"""阶段E7：CTP基础Profile与离线Bar模拟撮合测试。

E7a验证基础Profile和期货合约工厂；E7b使用market/replay读取rb2704 Feather，
验证合约乘数、固定每手手续费、保证金、基础开平仓和最终零仓位。

这里采用NETTING基础近似，不验证CTP双向持仓、今昨仓、平今/平昨手续费、
交易日切换和结算；这些能力属于E10。测试不连接CTP，也不会发送真实订单。
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.backtest.models import PerContractFeeModel, StandardMarginModel
from bomber.model import Money, Venue
from bomber.model.currencies import CNY
from bomber.model.data import Bar
from bomber.model.enums import AccountType, OmsType, PositionSide
from bomber.model.identifiers import InstrumentId, TraderId

from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import BarColumns, MappedBarParser
from strategy import (
    CtpFuturesBasicProfile,
    ExecutionReportType,
    NautilusSimExecutionBackend,
    OrderIntent,
    OrderSide,
    PositionEffect,
    VenueSimulationProfilePort,
)


BAR_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728/"
    "rb2704_20260728.feather",
)
INSTRUMENT_ID = InstrumentId.from_str("rb2704.SHFE")


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000_000)


def _profile() -> CtpFuturesBasicProfile:
    # 每手1元仅是本测试的显式输入，不代表交易所或期货公司的真实费率。
    return CtpFuturesBasicProfile(
        starting_balance=Decimal("1000000"),
        commission_per_contract=Decimal("1"),
        venue=Venue("SHFE"),
    )


def _instrument(profile: CtpFuturesBasicProfile):
    return profile.make_instrument(
        "rb2704",
        underlying="rb",
        price_precision=0,
        price_increment=Decimal("1"),
        multiplier=Decimal("10"),
        activation_ns=_ns("2026-01-01"),
        expiration_ns=_ns("2027-05-01"),
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.08"),
    )


def test1_ctp_basic_profile_and_instrument() -> None:
    """E7a：验证基础Venue规则和螺纹钢合约参数映射。"""

    profile = _profile()
    config = profile.build_backend_config()
    instrument = _instrument(profile)

    assert isinstance(profile, VenueSimulationProfilePort)
    assert profile.venue == Venue("SHFE")
    assert config["oms_type"] is OmsType.NETTING
    assert config["account_type"] is AccountType.MARGIN
    assert config["base_currency"] == CNY
    assert config["default_leverage"] == Decimal(1)
    assert isinstance(config["margin_model"], StandardMarginModel)
    assert isinstance(config["fee_model"], PerContractFeeModel)
    assert config["use_position_ids"] is False
    assert config["use_reduce_only"] is True

    assert instrument.id == INSTRUMENT_ID
    assert instrument.exchange == "SHFE"
    assert instrument.quote_currency == CNY
    assert instrument.price_precision == 0
    assert str(instrument.price_increment) == "1"
    assert instrument.multiplier.as_decimal() == Decimal(10)
    assert instrument.size_increment.as_decimal() == Decimal(1)
    assert instrument.margin_init == Decimal("0.10")
    assert instrument.margin_maint == Decimal("0.08")
    # fut_basic中的minChgPriceNum常以浮点1.0存储；不能让文本形式的
    # 尾随零把Nautilus Price精度抬成1，与合约价格精度0冲突。
    integer_tick = profile.make_instrument(
        "rb2705", underlying="rb", price_precision=0,
        price_increment=Decimal("1.0"), multiplier=Decimal("10"),
        activation_ns=_ns("2026-01-01"), expiration_ns=_ns("2027-05-01"),
        margin_init=Decimal("0.10"), margin_maint=Decimal("0.08"),
    )
    assert integer_tick.price_precision == 0
    assert integer_tick.price_increment.precision == 0
    assert str(integer_tick.price_increment) == "1"
    fractional_tick = profile.make_instrument(
        "rb2706", underlying="rb", price_precision=1,
        price_increment=Decimal("0.50"), multiplier=Decimal("10"),
        activation_ns=_ns("2026-01-01"), expiration_ns=_ns("2027-05-01"),
        margin_init=Decimal("0.10"), margin_maint=Decimal("0.08"),
    )
    assert fractional_tick.price_increment.precision == 1
    print("E7a通过：CTP基础Profile和rb期货合约规则映射正常")


def _load_bars() -> list[Bar]:
    feed = FileReplayFeed("E7_CTP_BAR_LOADER")
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=INSTRUMENT_ID,
            price_precision=0,
            size_precision=0,
            price_increment=Decimal(1),
            multiplier=Decimal(10),
            currency="CNY",
            exchange="SHFE",
        ),
    )
    feed.add_bar_feather(
        BAR_PATH,
        MappedBarParser(
            columns=BarColumns(
                symbol="symbol",
                exchange="exchange",
                timestamp="datetime",
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                value="value",
                open_interest="open_interest",
                vwap="vwap",
            ),
            bar_spec="1-MINUTE",
            timezone="Asia/Shanghai",
            exchange_aliases={"XSGE": "SHFE"},
        ),
    )
    feed.subscribe(INSTRUMENT_ID, DataType.BAR, bar_spec="1-MINUTE")
    feed.connect()
    try:
        return [event for event in feed.load_events() if isinstance(event, Bar)]
    finally:
        feed.disconnect()


def _commission_value(report) -> Decimal:
    raw = str(report.metadata["commission"]).replace("_", "")
    return Decimal(raw.split()[0])


def test2_ctp_bar_basic_round_trip() -> None:
    """E7b：真实CTP Bar完成基础开平仓、保证金和每手手续费验证。"""

    profile = _profile()
    instrument = _instrument(profile)
    bars = _load_bars()
    if len(bars) < 3:
        raise AssertionError("CTP离线Bar文件至少需要三根Bar")

    backend = NautilusSimExecutionBackend(
        "e7-ctp-basic",
        BacktestEngineConfig(trader_id=TraderId("E7-TESTER"), run_analysis=False),
    )
    backend.add_profile(profile)
    backend.add_instrument(instrument)
    try:
        # 第一根Bar启动模拟Venue，并建立可供保证金计算使用的账户和市场价格。
        backend.process_market_event(bars[0])
        account = backend.engine.portfolio.account(profile.venue)
        if account is None:
            raise AssertionError("未创建CTP模拟保证金账户")

        probe_quantity = instrument.make_qty(1)
        probe_price = instrument.make_price(3100)
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
        assert initial_margin.as_decimal() == Decimal("3100.00")
        assert maintenance_margin.as_decimal() == Decimal("2480.00")

        backend.submit_order(
            OrderIntent(
                strategy_id="e7-ctp-alpha",
                backend_id=backend.backend_id,
                instrument_id=instrument.id,
                side=OrderSide.BUY,
                quantity=1,
                position_effect=PositionEffect.OPEN,
                metadata={"action": "open"},
            ),
        )
        open_reports = backend.process_market_event(bars[1])
        open_fill = next(
            report
            for report in open_reports
            if report.report_type is ExecutionReportType.FILLED
        )
        if open_fill.fill_price is None:
            raise AssertionError("CTP开仓成交回报缺少成交价")
        assert open_fill.filled_quantity == Decimal(1)
        assert _commission_value(open_fill) == Decimal(1)
        expected_maintenance = (
            open_fill.fill_price
            * open_fill.filled_quantity
            * instrument.multiplier.as_decimal()
            * instrument.margin_maint
        )
        locked_maintenance = account.margin_maint(instrument.id)
        if locked_maintenance is None:
            raise AssertionError("CTP开仓后没有生成维持保证金")
        assert locked_maintenance.as_decimal() == expected_maintenance
        assert account.balance_locked(CNY).as_decimal() == expected_maintenance

        backend.submit_order(
            OrderIntent(
                strategy_id="e7-ctp-alpha",
                backend_id=backend.backend_id,
                instrument_id=instrument.id,
                side=OrderSide.SELL,
                quantity=1,
                position_effect=PositionEffect.CLOSE,
                metadata={"action": "close-basic"},
            ),
        )
        close_reports = backend.process_market_event(bars[2])
        close_fill = next(
            report
            for report in close_reports
            if report.report_type is ExecutionReportType.FILLED
        )
        assert _commission_value(close_fill) == Decimal(1)
        assert close_fill.metadata["position_effect"] == PositionEffect.CLOSE.value
        assert account.margin_maint(instrument.id) is None
        assert account.balance_locked(CNY).as_decimal() == 0
        assert Decimal(str(backend.engine.portfolio.net_position(instrument.id))) == 0

        native_orders = backend.engine.cache.orders()
        assert len(native_orders) == 2
        assert sum(order.is_reduce_only for order in native_orders) == 1
    finally:
        backend.stop()

    result = backend.result()
    assert result is not None
    assert result.total_orders == 2
    print(
        "E7b通过：CTP Feather Bar完成基础开仓、CLOSE平仓、每手手续费和保证金验证，"
        f"bars={len(bars):,} orders={result.total_orders}",
    )


STAGES = {
    1: test1_ctp_basic_profile_and_instrument,
    2: test2_ctp_bar_basic_round_trip,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CTP基础模拟执行分阶段测试")
    parser.add_argument("--stage", choices=(*(str(i) for i in STAGES), "all"), default="all")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {int(args.stage): STAGES[int(args.stage)]}
    for function in selected.values():
        function()


if __name__ == "__main__":
    main()
