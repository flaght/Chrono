"""CTP五品种或Binance五品种1分钟Bar离线回测。

CTP Tick入口保留给后续真实Tick文件；当前指定的十份文件均为Bar。
合约精度、乘数、费用和保证金是示例参数，正式绩效评估前须用历史规则核准。
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.model import Venue
from bomber.model.identifiers import InstrumentId, TraderId
from bomber.model.objects import Currency


from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import (
    BarColumns,
    BinanceKlineParser,
    BinanceMarketType,
    MappedBarParser
)
from strategy import (
    CtpFuturesBasicProfile,
    BinanceUsdtFuturesProfile,
    DataBinding,
    ExecutionRoute,
    MarketReferencePriceStore,
    MarketStreamBinding,
    NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend,
    NetTargetOrderPlanner,
    PositionManager,
    PreTradeRiskManager,
    RiskLimits,
    RuntimeMode,
    SimulationExecutionClient,
    UnifiedHistoricalRuntime,
    UnifiedStrategyRunner,
)


from examples.cross_section.cross_strategy import CrossSectionConfig, CrossSectionMomentumStrategy

def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000_000)


@dataclass(frozen=True)
class ContractSpec:
    root: str
    symbol: str
    venue: str
    multiplier: int
    price_increment: Decimal
    price_precision: int
    filename: str

    @property
    def instrument_id(self) -> InstrumentId:
        return InstrumentId.from_str(f"{self.symbol}.{self.venue}")



CONTRACT_SPECS = (
    ContractSpec("RB", "rb2704", "SHFE", 10, Decimal(1),
                 0, "rb2704_20260728.feather"),
    ContractSpec("SA", "SA703", "CZCE", 20, Decimal(1),
                 0, "SA703_20260728.feather"),
    ContractSpec("MA", "MA704", "CZCE", 10, Decimal(1),
                 0, "MA704_20260728.feather"),
    ContractSpec("HC", "hc2702", "SHFE", 10, Decimal(1),
                 0, "hc2702_20260728.feather"),
    ContractSpec("JM", "jm2608", "DCE", 60, Decimal(
        "0.5"), 1, "jm2608_20260728.feather"),
)

BINANCE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "JUPUSDT", "LQTYUSDT", "APRUSDT")
BN_PRICE_INCREMENT = Decimal("0.00000001")
BN_SIZE_INCREMENT = Decimal("0.00000001")


CTP_DATA_DIR = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728")
BINANCE_DATA_DIR = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/binance_data/futures/um/klines/1m"
)


def create_instrument(instrument_name: str):
    """统一构造Profile、可交易合约和行情解析元数据。

    与single_ema示例相同：Profile给Backend设置Venue规则，instrument用于撮合，
    InstrumentMeta供Replay Parser按同一合约精度生成标准行情。
    """
    spec = next(
        (item for item in CONTRACT_SPECS if item.symbol == instrument_name), None)
    if spec is not None:
        profile = CtpFuturesBasicProfile(
            profile_id=f"cross-section-{spec.venue.lower()}",
            starting_balance=Decimal("1000000"),
            commission_per_contract=Decimal(1),
            venue=Venue(spec.venue),
        )
        instrument = profile.make_instrument(
            spec.symbol,
            underlying=spec.root,
            price_precision=spec.price_precision,
            price_increment=spec.price_increment,
            multiplier=spec.multiplier,
            activation_ns=_ns("2025-01-01"),
            expiration_ns=_ns("2027-12-31"),
            margin_init=Decimal("0.12"),
            margin_maint=Decimal("0.12"),
        )
        meta = InstrumentMeta(
            instrument.id,
            price_precision=spec.price_precision,
            size_precision=0,
            price_increment=spec.price_increment,
            multiplier=Decimal(spec.multiplier),
            currency="CNY",
            exchange=spec.venue,
        )
        return profile, instrument, meta

    if instrument_name in BINANCE_SYMBOLS:
        profile = BinanceUsdtFuturesProfile(
            profile_id="cross-section-binance",
            starting_balance=Decimal("1000000"),
        )
        instrument = profile.make_instrument(
            instrument_name,
            price_precision=8,
            size_precision=8,
            price_increment=BN_PRICE_INCREMENT,
            size_increment=BN_SIZE_INCREMENT,
            base_currency=Currency.from_str(
                instrument_name.removesuffix("USDT")),
            min_quantity=BN_SIZE_INCREMENT,
        )
        meta = InstrumentMeta(
            instrument.id,
            price_precision=8,
            size_precision=8,
            price_increment=BN_PRICE_INCREMENT,
            multiplier=Decimal(1),
            currency="USDT",
            exchange="BINANCE",
        )
        return profile, instrument, meta
    raise ValueError(f"不支持的回测合约: {instrument_name}")


def build_feed(
    data_dir: Path,
    *,
    market: str = "ctp"
):
    feed = FileReplayFeed("CROSS_SECTION_BAR_REPLAY")
    if market == "binance":
        for symbol in BINANCE_SYMBOLS:
            path = data_dir / symbol / "2025-10-23.csv"
            _, _, meta = create_instrument(symbol)
            feed.register_instrument(meta)
            feed.add_bar_csv(path, BinanceKlineParser(
                symbol=symbol,
                market_type=BinanceMarketType.FUTURES,
                interval="1m",
                include_factors=False,
            ))
    elif market == "ctp":
        parser = MappedBarParser(
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
            exchange_aliases={"XSGE": "SHFE", "XZCE": "CZCE", "XDCE": "DCE"},
        )
        for spec in CONTRACT_SPECS:
            path = data_dir / f"{spec.symbol}_20260728.feather"
            _, _, meta = create_instrument(spec.symbol)
            feed.register_instrument(meta)
            feed.add_bar_feather(path, parser)
    else:
        raise ValueError(f"不支持的行情市场: {market}")
    return feed


def run_case(
    data_dir: Path,
    market: str = "ctp",
    lookback: int = 20,
    rebalance_interval: int = 5,
    target_notional: Decimal = Decimal("100000"),
    night_session_action_day: str | None = None,
) -> None:
    feed = build_feed(
        data_dir=data_dir, market=market
    )
    backend = NautilusSimExecutionBackend(
        f"cross-section-{market}-sim",
        BacktestEngineConfig(trader_id=TraderId(
            "CROSS-SECTION-001"), run_analysis=True),
    )
    if market == "ctp":
        instruments = tuple(spec.instrument_id for spec in CONTRACT_SPECS)
        multipliers = {spec.instrument_id: Decimal(
            spec.multiplier) for spec in CONTRACT_SPECS}
        steps = {item: Decimal(1) for item in instruments}
        names = tuple(spec.symbol for spec in CONTRACT_SPECS)
    else:
        instruments = tuple(create_instrument(symbol)[1].id for symbol in BINANCE_SYMBOLS)
        multipliers = {item: Decimal(1) for item in instruments}
        steps = {item: BN_SIZE_INCREMENT for item in instruments}
        names = BINANCE_SYMBOLS

    # 一个Venue只注册一次Profile；每个instrument和Feed中的Meta均由同一工厂生成。
    registered_venues: set[str] = set()
    for name in names:
        profile, instrument, _ = create_instrument(name)
        venue_key = str(profile.venue)
        if venue_key not in registered_venues:
            backend.add_profile(profile)
            registered_venues.add(venue_key)
        backend.add_instrument(instrument)

    positions = PositionManager()
    prices = MarketReferencePriceStore()

    client = SimulationExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id,
            positions,
            prices,
            instrument_limits={
                item: RiskLimits(
                    max_order_notional=Decimal("250000"),
                    max_abs_position_notional=Decimal("250000"),
                    max_market_age_ns=120 * 1_000_000_000,
                    contract_multiplier=multipliers[item],
                )
                for item in instruments
            },
        ),
    )

    strategy = CrossSectionMomentumStrategy(
        f"cross-section-{market}",
        CrossSectionConfig(
            instruments=instruments,
            contract_multipliers=multipliers,
            size_increments=steps,
            lookback=lookback,
            rebalance_interval=rebalance_interval,
            target_notional=target_notional,
        ),
    )

    market_adapter = NautilusMarketFeedAdapter(
        f"cross-section-{market}-clock",
        feed,
        backend,
        tuple(
            binding
            for item in instruments
            for binding in (
                (
                    MarketStreamBinding(item, DataType.BAR, "1-MINUTE"),
                )
            )
        ),
        manage_lifecycle=False,
    )

    runner = UnifiedStrategyRunner(
        RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("cross-bars", feed)
    runner.add_execution_client(client)

    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "cross-bars", item,
                        DataType.BAR, "1-MINUTE")
            for item in instruments
        ),
        execution_routes=tuple(
            ExecutionRoute(str(item), backend.backend_id, item)
            for item in instruments
        ),
    )

    runtime = UnifiedHistoricalRuntime(
        f"cross-section-{market}-runtime", runner, market_adapter)
    try:
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        print(
            f"[{market}] raw_events={result.replay_summary.total:,} "
            f"complete_frames={strategy.synchronized_frames:,} "
            f"rebalances={strategy.rebalances:,} "
            f"orders={len(orders)} fills={len(fills)}",
        )
        print(
            f"[{market}] final_targets={dict(strategy.last_targets or {})}")
        if strategy.rebalances < 1 or strategy.last_targets is None:
            raise AssertionError("没有形成五品种截面目标，请检查样本是否包含足够完整同步帧")
        if not orders.empty and fills.empty:
            raise AssertionError("已生成订单但没有模拟成交")
        if client.report_errors:
            raise AssertionError(f"统一回报状态异常: {client.report_errors}")
    finally:
        runtime.stop()

def main() -> None:
    run_case(data_dir=BINANCE_DATA_DIR, market="binance")
    #run_case(data_dir=CTP_DATA_DIR, market="ctp")


if __name__ == "__main__":
    main()
