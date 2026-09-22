"""第五类正式基础回测：真实JM/I/RB Bar -> 复权信号 -> RB主力原生撮合。

仅示范CTP Basic Profile；不声称覆盖交易所平今/平昨与结算高保真规则。
Feather读取和本地路径只在装配层，策略与DataHub内核不依赖存储格式。
"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
from bomber.backtest.config import BacktestEngineConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import FixedInstrumentBarParser
from strategy import (
    ContractAssignment, CtpFuturesBasicProfile, DataBinding, DynamicExecutionRoute,
    MarketReferencePriceStore, MarketStreamBinding, NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend, NetTargetOrderPlanner, PositionManager,
    PreTradeRiskManager, RiskLimits, RuntimeMode, ScheduledContractResolver,
    SimulationExecutionClient, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)

from examples.black_sector.local_input import load_sector_research
from examples.black_sector.sector_strategy import BlackSectorConfig, BlackSectorTargetStrategy


ROLE_ROOT = Path("/workspace/worker/pj/neutron/tests/temp/role")
BARS_ROOT = Path("/workspace/data/dev/kd/intelkit/records/temp")
DAY_NS = 86_400_000_000_000


def _day_ns(value: object) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("合约挂牌/到期日不能为空")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.tz_convert("UTC").value)


def _bar_path(root: Path, symbol: str, day: date) -> Path:
    matches = tuple(path for path in root.rglob(f"*_{day:%Y%m%d}.feather")
                    if path.name.lower() == f"{symbol.lower()}_{day:%Y%m%d}.feather")
    if len(matches) != 1:
        raise FileNotFoundError(f"{day}/{symbol}需要唯一真实合约Bar，找到{len(matches)}个")
    return matches[0]


def _timestamp_column(path: Path) -> str:
    import pyarrow as pa

    for key in ("datetime", "timestamp"):
        try:
            pd.read_feather(path, columns=[key])
        except (KeyError, ValueError, pa.ArrowInvalid):
            continue
        return key
    raise ValueError(f"{path}缺少datetime/timestamp")


def _instrument(basic: pd.DataFrame, profiles: dict[str, CtpFuturesBasicProfile],
                symbol: str, price_increment: Decimal):
    selected = basic.loc[basic["symbol"].astype(str).str.strip().str.lower() == symbol]
    if len(selected) != 1:
        raise ValueError(f"fut_basic需要唯一合约记录: {symbol}，实际{len(selected)}")
    row = selected.iloc[0]
    exchange = str(row["exchangeCD"]).upper()
    venue = {"XSGE": "SHFE", "SHFE": "SHFE", "XDCE": "DCE", "DCE": "DCE"}.get(exchange)
    if venue is None:
        raise ValueError(f"暂不支持的交易所: {symbol}/{exchange}")
    multiplier = Decimal(str(row["contMultNum"]))
    if not multiplier.is_finite() or multiplier <= 0:
        raise ValueError(f"合约乘数无效: {symbol}")
    # 有交易所合约最小变动价位时优先采用；命令行值仅为旧元数据缺列的回退。
    if "minChgPriceNum" in row and pd.notna(row["minChgPriceNum"]):
        price_increment = Decimal(str(row["minChgPriceNum"]))
    if not price_increment.is_finite() or price_increment <= 0:
        raise ValueError(f"最小变动价位无效: {symbol}")
    precision = max(0, -price_increment.normalize().as_tuple().exponent)
    instrument = profiles[venue].make_instrument(
        symbol, underlying=symbol.rstrip("0123456789"),
        price_precision=precision, price_increment=price_increment,
        multiplier=multiplier, activation_ns=_day_ns(row["listDate"]),
        expiration_ns=_day_ns(row["lastTradeDate"]) + DAY_NS,
        margin_init=Decimal("0.10"), margin_maint=Decimal("0.08"),
    )
    meta = InstrumentMeta(
        instrument.id, price_precision=precision, size_precision=0,
        price_increment=price_increment, multiplier=multiplier,
        currency="CNY", exchange=venue,
    )
    return instrument, meta, multiplier


def run_case(*, start_day: date, end_day: date, bars_dir: Path,
             contract_struct: Path, fut_basic: Path,
             price_increment: Decimal, quantity: Decimal = Decimal(1),
             submission_delay_bars: int = 0) -> None:
    if end_day < start_day or not price_increment.is_finite() or price_increment <= 0:
        raise ValueError("日期范围或价格步长无效")
    config = BlackSectorConfig(quantity=quantity, submission_delay_bars=submission_delay_bars)
    loaded = load_sector_research(
        bars_dir=bars_dir, contract_struct_path=contract_struct,
        signal_products=config.signal_products, signal_role=config.signal_role,
        execution_product=config.execution_product, execution_role=config.execution_role,
    )
    ends = dict(loaded.day_end_ns)
    days = tuple(day for day in sorted(ends) if start_day <= day <= end_day)
    if not days:
        raise ValueError("所选区间没有JM/I/RB完整角色与真实Bar")
    assignments = tuple(loaded.store.snapshot(ends[day]) for day in days)
    symbols = tuple(sorted({symbol for row in assignments for symbol in (
        *(row.instrument(product, config.signal_role) for product in config.signal_products),
        row.instrument(config.execution_product, config.execution_role),
    )}))
    basic = pd.read_feather(fut_basic)
    required = {"symbol", "exchangeCD", "contMultNum", "listDate", "lastTradeDate"}
    if required - set(basic.columns):
        raise ValueError(f"fut_basic缺少列: {sorted(required - set(basic.columns))}")

    backend = NautilusSimExecutionBackend(
        "black-sector-sim",
        BacktestEngineConfig(trader_id=TraderId("BLACK-SECTOR-001"), run_analysis=True),
    )
    profiles = {
        venue: CtpFuturesBasicProfile(starting_balance=Decimal("1000000"),
                                      commission_per_contract=Decimal(1), venue=Venue(venue))
        for venue in ("SHFE", "DCE")
    }
    for profile in profiles.values():
        backend.add_profile(profile)
    feed = FileReplayFeed("BLACK_SECTOR_REPLAY")
    instruments = {}
    multipliers = {}
    for symbol in symbols:
        instrument, meta, multiplier = _instrument(basic, profiles, symbol, price_increment)
        instruments[symbol] = instrument.id
        multipliers[instrument.id] = multiplier
        backend.add_instrument(instrument)
        feed.register_instrument(meta)
    for index, (day, assignment) in enumerate(zip(days, assignments)):
        execution_symbol = assignment.instrument(config.execution_product, config.execution_role)
        needed = {assignment.instrument(product, config.signal_role)
                  for product in config.signal_products} | {execution_symbol}
        if index and assignments[index - 1].instrument(
            config.execution_product, config.execution_role,
        ) != execution_symbol:
            # 换月旧主力仍需当日真实Bar，才能验证先平旧仓。
            needed.add(assignments[index - 1].instrument(
                config.execution_product, config.execution_role,
            ))
        for symbol in sorted(needed):
            path = _bar_path(bars_dir, symbol, day)
            feed.add_bar_feather(
                path, FixedInstrumentBarParser(instruments[symbol], timestamp=_timestamp_column(path)),
            )
    resolver = ScheduledContractResolver(tuple(
        ContractAssignment(config.target_key,
                           instruments[row.instrument(config.execution_product, config.execution_role)],
                           row.effective_ns,
                           row.available_ns, revision)
        for revision, row in enumerate(assignments, 1)
    ))
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    client = SimulationExecutionClient(
        backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id, positions, prices,
            instrument_limits={item: RiskLimits(
                max_order_quantity=Decimal(2), max_abs_position=Decimal(1),
                max_order_notional=Decimal("1000000"),
                max_abs_position_notional=Decimal("1000000"),
                max_market_age_ns=120 * 1_000_000_000,
                contract_multiplier=multipliers[item],
            ) for item in multipliers},
        ),
    )
    strategy = BlackSectorTargetStrategy(
        "black-sector-formal", loaded.store, config,
    )
    adapter = NautilusMarketFeedAdapter(
        "black-sector-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE") for item in instruments.values()),
        manage_lifecycle=False,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("sector-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(DataBinding(str(item), "sector-bars", item, DataType.BAR, "1-MINUTE")
                            for item in instruments.values()),
        execution_routes=(DynamicExecutionRoute(config.target_key, backend.backend_id, resolver),),
    )
    runtime = UnifiedHistoricalRuntime("black-sector-runtime", runner, adapter)
    try:
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        print(f"V4正式基础回测: days={len(days)} bars={result.replay_summary.bars} "
              f"complete_frames={strategy.complete_frames} signals={len(strategy.signal_events)} "
              f"targets={sum(item['kind'] == 'target_submitted' for item in strategy.execution_events)} "
              f"orders={len(orders)} fills={len(fills)} unavailable={strategy.unavailable_events}")
        for item in strategy.execution_events[-10:]:
            print(f"V4目标审计: {item}")
    finally:
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-day", type=date.fromisoformat, default=date(2026, 1, 5))
    parser.add_argument("--end-day", type=date.fromisoformat, default=date(2026, 2, 26))
    parser.add_argument("--bars-dir", type=Path, default=BARS_ROOT)
    parser.add_argument("--contract-struct", type=Path, default=ROLE_ROOT / "fut_contract_data.feather")
    parser.add_argument("--fut-basic", type=Path, default=ROLE_ROOT / "fut_basic.feather")
    parser.add_argument("--price-increment", type=Decimal, default=Decimal("0.01"))
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--submission-delay-bars", type=int, default=0)
    args = parser.parse_args()
    run_case(start_day=args.start_day, end_day=args.end_day, bars_dir=args.bars_dir,
             contract_struct=args.contract_struct, fut_basic=args.fut_basic,
             price_increment=args.price_increment, quantity=args.quantity,
             submission_delay_bars=args.submission_delay_bars)


if __name__ == "__main__":
    main()
