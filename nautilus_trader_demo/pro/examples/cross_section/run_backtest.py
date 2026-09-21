"""第二类策略：CTP五品种或Binance五品种1分钟Bar离线回测。

CTP Tick入口保留给后续真实Tick文件；当前指定的十份文件均为Bar。
合约精度、乘数、费用和保证金是示例参数，正式绩效评估前须用历史规则核准。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.model import Venue
from bomber.model.identifiers import InstrumentId, TraderId
from bomber.model.objects import Currency

from examples.cross_section.strategy import CrossSectionConfig, CrossSectionMomentumStrategy
from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import BinanceKlineParser, BinanceMarketType, FixedInstrumentBarParser
from market.replay.parsers.tick import CtpQuoteParser
from market.stream import QuoteMidBarFeed
from strategy.bar_sync import BarSynchronizer
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
    ContractSpec("RB", "rb2704", "SHFE", 10, Decimal(1), 0, "rb2704_20260728.feather"),
    ContractSpec("SA", "SA703", "CZCE", 20, Decimal(1), 0, "SA703_20260728.feather"),
    ContractSpec("MA", "MA704", "CZCE", 10, Decimal(1), 0, "MA704_20260728.feather"),
    ContractSpec("HC", "hc2702", "SHFE", 10, Decimal(1), 0, "hc2702_20260728.feather"),
    ContractSpec("JM", "jm2608", "DCE", 60, Decimal("0.5"), 1, "jm2608_20260728.feather"),
)

CTP_DATA_DIR = Path("/workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728")
BINANCE_DATA_DIR = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/binance_data/futures/um/klines/1m"
)
BINANCE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "JUPUSDT", "LQTYUSDT", "APRUSDT")
BN_PRICE_INCREMENT = Decimal("0.00000001")
BN_SIZE_INCREMENT = Decimal("0.00000001")


def _binance_id(symbol: str) -> InstrumentId:
    return InstrumentId.from_str(f"{symbol}-PERP.BINANCE")


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000_000)


def create_instrument(instrument_name: str):
    """统一构造Profile、可交易合约和行情解析元数据。

    与single_ema示例相同：Profile给Backend设置Venue规则，instrument用于撮合，
    InstrumentMeta供Replay Parser按同一合约精度生成标准行情。
    """
    spec = next((item for item in CONTRACT_SPECS if item.symbol == instrument_name), None)
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
            base_currency=Currency.from_str(instrument_name.removesuffix("USDT")),
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
    source: str,
    data_dir: Path,
    *,
    market: str = "ctp",
    night_session_action_day: str | None = None,
):
    """Tick仅解析一档Quote并聚合MID Bar；Feather直接解析外部Bar。"""
    if source not in {"tick", "bar"}:
        raise ValueError(f"不支持的离线来源: {source}")
    if market not in {"ctp", "binance"} or (market == "binance" and source == "tick"):
        raise ValueError("Binance当前只支持指定的1分钟Bar；Tick入口仅支持CTP")
    upstream = FileReplayFeed(f"CROSS_SECTION_{source.upper()}_REPLAY")
    feed = QuoteMidBarFeed("CROSS_SECTION_MID_BARS", upstream) if source == "tick" else upstream
    if market == "binance":
        for symbol in BINANCE_SYMBOLS:
            path = data_dir / symbol / "2023-09-02.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            _, _, meta = create_instrument(symbol)
            feed.register_instrument(meta)
            upstream.add_bar_csv(path, BinanceKlineParser(
                symbol, BinanceMarketType.FUTURES, interval="1m", include_factors=False,
            ))
        return feed
    for spec in CONTRACT_SPECS:
        exact_path = data_dir / (spec.filename if source == "bar" else spec.filename.replace(".feather", ".csv"))
        files = [exact_path] if exact_path.is_file() else sorted((data_dir / spec.root).glob(f"*{suffix}"))
        if not files:
            raise FileNotFoundError(f"缺少{spec.root}文件: {exact_path}")
        _, _, meta = create_instrument(spec.symbol)
        feed.register_instrument(meta)
        for path in files:
            if source == "tick":
                upstream.add_tick_csv(
                    path,
                    CtpQuoteParser(
                        spec.venue,
                        night_session_action_day=night_session_action_day,
                        expected_symbol=spec.symbol,
                    ),
                )
            else:
                upstream.add_bar_feather(path, FixedInstrumentBarParser(spec.instrument_id))
    return feed


def run_feed_probe(market: str, source: str, data_dir: Path) -> None:
    """第一步只验证五个文件能解析并形成同时间戳截面，不启动撮合。"""
    feed = build_feed(source, data_dir, market=market)
    instruments = (
        tuple(spec.instrument_id for spec in CONTRACT_SPECS)
        if market == "ctp" else tuple(_binance_id(symbol) for symbol in BINANCE_SYMBOLS)
    )
    synchronizer = BarSynchronizer(instruments)
    counts = {item: 0 for item in instruments}
    frames = []

    def on_bar(bar) -> None:
        item = bar.bar_type.instrument_id
        counts[item] += 1
        frame = synchronizer.push(bar)
        if frame is not None:
            frames.append(frame.ts_event)

    feed.register_bar_handler(on_bar)
    for item in instruments:
        feed.subscribe(item, DataType.BAR, "1-MINUTE")
    feed.connect()
    try:
        summary = feed.replay()
    finally:
        feed.disconnect()
    print(f"[{market}/{source}] raw_events={summary.total:,} bars_per_instrument={counts}")
    print(f"[{market}/{source}] complete_frames={len(frames)} first={frames[0] if frames else None} last={frames[-1] if frames else None}")
    if any(count == 0 for count in counts.values()):
        raise AssertionError("至少一个标的没有解析出标准Bar")
    if not frames:
        raise AssertionError("五个标的没有共同的1分钟时间戳，不能执行截面策略")


def run_case(
    source: str,
    data_dir: Path,
    *,
    market: str = "ctp",
    lookback: int = 20,
    rebalance_interval: int = 5,
    target_notional: Decimal = Decimal("100000"),
    night_session_action_day: str | None = None,
) -> None:
    """两种输入只替换Feed，截面策略/目标/执行链保持一致。"""
    feed = build_feed(
        source, data_dir, market=market,
        night_session_action_day=night_session_action_day,
    )
    backend = NautilusSimExecutionBackend(
        f"cross-section-{market}-{source}-sim",
        BacktestEngineConfig(trader_id=TraderId("CROSS-SECTION-001"), run_analysis=True),
    )
    if market == "ctp":
        instruments = tuple(spec.instrument_id for spec in CONTRACT_SPECS)
        multipliers = {spec.instrument_id: Decimal(spec.multiplier) for spec in CONTRACT_SPECS}
        steps = {item: Decimal(1) for item in instruments}
        names = tuple(spec.symbol for spec in CONTRACT_SPECS)
    else:
        instruments = tuple(_binance_id(symbol) for symbol in BINANCE_SYMBOLS)
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
        f"cross-section-{market}-{source}",
        CrossSectionConfig(
            instruments=instruments,
            contract_multipliers=multipliers,
            size_increments=steps,
            lookback=lookback,
            rebalance_interval=rebalance_interval,
            target_notional=target_notional,
        ),
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("five-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "five-bars", item, DataType.BAR, "1-MINUTE")
            for item in instruments
        ),
        execution_routes=tuple(
            ExecutionRoute(str(item), backend.backend_id, item)
            for item in instruments
        ),
    )
    adapter = NautilusMarketFeedAdapter(
        f"cross-section-{market}-{source}-clock",
        feed,
        backend,
        tuple(
            binding
            for item in instruments
            for binding in (
                (
                    MarketStreamBinding(item, DataType.QUOTE_TICK),
                    MarketStreamBinding(item, DataType.BAR, "1-MINUTE"),
                )
                if source == "tick"
                else (MarketStreamBinding(item, DataType.BAR, "1-MINUTE"),)
            )
        ),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime(f"cross-section-{market}-{source}-runtime", runner, adapter)
    try:
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        print(
            f"[{market}/{source}] raw_events={result.replay_summary.total:,} "
            f"complete_frames={strategy.synchronized_frames:,} "
            f"rebalances={strategy.rebalances:,} "
            f"orders={len(orders)} fills={len(fills)}",
        )
        print(f"[{market}/{source}] final_targets={dict(strategy.last_targets or {})}")
        if strategy.rebalances < 1 or strategy.last_targets is None:
            raise AssertionError("没有形成五品种截面目标，请检查样本是否包含足够完整同步帧")
        if not orders.empty and fills.empty:
            raise AssertionError("已生成订单但没有模拟成交")
        if client.report_errors:
            raise AssertionError(f"统一回报状态异常: {client.report_errors}")
    finally:
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="第二类五品种截面动量离线正式回测")
    parser.add_argument("--market", choices=("ctp", "binance"), default="ctp")
    parser.add_argument("--stage", choices=("feed", "formal"), default="feed")
    parser.add_argument("--source", choices=("bar", "tick"), default="bar")
    parser.add_argument("--data-dir", type=Path, help="默认使用用户指定的2026-07-28 CTP或2023-09-02 BN目录")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--target-notional", type=Decimal, default=Decimal("100000"))
    parser.add_argument("--night-session-action-day", help="Tick夜盘自然日，例如20260727")
    args = parser.parse_args()
    data_dir = args.data_dir or (CTP_DATA_DIR if args.market == "ctp" else BINANCE_DATA_DIR)
    if args.stage == "feed":
        run_feed_probe(args.market, args.source, data_dir)
    else:
        run_case(
            args.source,
            data_dir,
            market=args.market,
            lookback=args.lookback,
            rebalance_interval=args.rebalance_interval,
            target_notional=args.target_notional,
            night_session_action_day=args.night_session_action_day,
        )


if __name__ == "__main__":
    main()
