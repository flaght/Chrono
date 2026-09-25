"""CTP 黑色系次主力信号、主力执行的可配置离线回测。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv 
load_dotenv()


import pandas as pd
from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from demos.black_sector.local_input import load_sector_research
from demos.black_sector.sector_strategy import BlackSectorConfig, BlackSectorTargetStrategy
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

VENUES = {"XSGE": "SHFE", "SHFE": "SHFE", "XDCE": "DCE", "DCE": "DCE",
          "XZCE": "CZCE", "CZCE": "CZCE", "XSIE": "INE", "INE": "INE"}
DAY_NS = 86_400_000_000_000
MAX_COVERAGE_GAP = timedelta(days=14)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"


def _day_ns(value: object) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("合约挂牌/到期日不能为空")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.tz_convert("UTC").value)


def _bar_paths(root: Path, required: set[tuple[date, str]]) -> dict[tuple[date, str], Path]:
    """一次扫描建立所需 Bar 索引，避免每个交易日重复遍历整个目录。"""
    expected = {f"{symbol.lower()}_{day:%Y%m%d}.feather": (day, symbol)
                for day, symbol in required}
    found = {}
    for path in root.rglob("*.feather"):
        key = expected.get(path.name.lower())
        if key is None:
            continue
        if key in found:
            raise ValueError(f"{key[0]}/{key[1]} 有多个真实合约 Bar: {found[key]}, {path}")
        found[key] = path
    missing = required - found.keys()
    if missing:
        day, symbol = min(missing)
        raise FileNotFoundError(f"{day}/{symbol} 缺少真实合约 Bar: {root}")
    return found


def _timestamp_column(path: Path) -> str:
    import pyarrow as pa
    for name in ("datetime", "timestamp"):
        try:
            pd.read_feather(path, columns=[name])
        except (KeyError, ValueError, pa.ArrowInvalid):
            continue
        return name
    raise ValueError(f"{path} 缺少 datetime/timestamp 列")


def _instrument(basic: pd.DataFrame, profile: CtpFuturesBasicProfile,
                product: str, symbol: str, margin_init: Decimal,
                margin_maint: Decimal):
    rows = basic.loc[basic["symbol"].astype(str).str.strip().str.lower() == symbol.lower()]
    if len(rows) != 1:
        raise ValueError(f"fut_basic 需要唯一合约记录: {symbol}，实际 {len(rows)}")
    row = rows.iloc[0]
    if str(row["code"]).strip().upper() != product or VENUES.get(
        str(row["exchangeCD"]).strip().upper()
    ) != str(profile.venue):
        raise ValueError(f"{symbol} 的品种或交易所与角色表不匹配")
    multiplier = Decimal(str(row["contMultNum"]))
    increment = Decimal(str(row["minChgPriceNum"]))
    if not multiplier.is_finite() or multiplier <= 0 or not increment.is_finite() or increment <= 0:
        raise ValueError(f"{symbol} 的 contMultNum/minChgPriceNum 无效")
    precision = max(0, -increment.normalize().as_tuple().exponent)
    instrument = profile.make_instrument(
        symbol, underlying=product.lower(), price_precision=precision,
        price_increment=increment, multiplier=multiplier,
        activation_ns=_day_ns(row["listDate"]),
        expiration_ns=_day_ns(row["lastTradeDate"]) + DAY_NS,
        margin_init=margin_init, margin_maint=margin_maint,
    )
    meta = InstrumentMeta(
        instrument.id, price_precision=precision, size_precision=0,
        price_increment=increment, multiplier=multiplier,
        currency="CNY", exchange=str(profile.venue),
    )
    return instrument, meta, multiplier


def _validate_basic(basic: pd.DataFrame, products: tuple[str, ...]) -> dict[str, str]:
    required = {"symbol", "code", "exchangeCD", "contMultNum", "minChgPriceNum",
                "listDate", "lastTradeDate"}
    if required - set(basic.columns):
        raise ValueError(f"fut_basic 缺少列: {sorted(required - set(basic.columns))}")
    venues = {}
    for product in products:
        rows = basic.loc[basic["code"].astype(str).str.strip().str.upper() == product]
        if rows.empty:
            raise ValueError(f"fut_basic 没有品种 {product}")
        candidates = {VENUES.get(str(item).strip().upper()) for item in rows["exchangeCD"]}
        if None in candidates or len(candidates) != 1:
            raise ValueError(f"{product} 的交易所不受支持或不唯一: {candidates}")
        venues[product] = candidates.pop()
    return venues


def _report(*, root: Path, backend, result, strategy: BlackSectorTargetStrategy,
            config: BlackSectorConfig, days: tuple[date, ...], requested: tuple[date, date],
            venues: set[str], orders: pd.DataFrame, fills: pd.DataFrame,
            tearsheet: bool) -> None:
    output = root / str(result.run_id)
    output.mkdir(parents=True, exist_ok=True)
    trader = backend.engine.trader
    orders.to_csv(output / "orders.csv")
    fills.to_csv(output / "fills.csv")
    trader.generate_positions_report().to_csv(output / "positions.csv")
    for venue in sorted(venues):
        trader.generate_account_report(Venue(venue)).to_csv(output / f"account_{venue}.csv")
    summary = asdict(result) if is_dataclass(result) else {"backend_result": str(result)}
    summary["strategy"] = {
        "leader_products": config.leader_products,
        "comparison_product": config.comparison_product,
        "execution_product": config.execution_product,
        "signal_role": config.signal_role,
        "execution_role": config.execution_role,
        "requested": [str(day) for day in requested],
        "actual": [str(days[0]), str(days[-1])],
        "complete_frames": strategy.complete_frames,
        "signals": len(strategy.signal_events),
        "targets": sum(event["kind"] == "target_submitted" for event in strategy.execution_events),
        "unavailable_events": strategy.unavailable_events,
        "execution_events": strategy.execution_events,
        "orders": len(orders), "fills": len(fills),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8",
    )
    print(f"回测报表: {output.resolve()}")
    if tearsheet:
        try:
            from bomber.analysis.tearsheet import create_tearsheet
            chart = output / "tearsheet.html"
            create_tearsheet(engine=backend.engine, output_path=str(chart),
                             title="CTP Black Sector Backtest")
        except ImportError as exc:
            print(f"绩效图未生成（{exc}）；CSV 和 JSON 已保存")
        else:
            print(f"交互绩效图: {chart.resolve()}")


def run_case(*, start_day: date, end_day: date, bars_dir: Path,
             contract_struct: Path, fut_basic: Path,
             leader_products: tuple[str, str] = ("JM", "I"),
             comparison_product: str = "RB", execution_product: str = "RB",
             signal_role: str = "secondary", execution_role: str = "main",
             quantity: Decimal = Decimal(1), return_period: int = 30,
             sector_period: int = 15, submission_delay_bars: int = 0,
             starting_balance: Decimal = Decimal("1000000"),
             commission_per_contract: Decimal = Decimal(1),
             margin_init: Decimal = Decimal("0.10"), margin_maint: Decimal = Decimal("0.08"),
             max_notional: Decimal = Decimal("1000000"),
             max_market_age_seconds: int = 120,
             log_level: str = "WARNING", report_dir: Path = DEFAULT_REPORT_DIR,
             tearsheet: bool = True, require_fills: bool = False) -> None:
    started_at = perf_counter()
    if end_day < start_day:
        raise ValueError("日期范围无效")
    if any(not item.is_finite() or item <= 0 for item in
           (starting_balance, margin_init, margin_maint, max_notional)):
        raise ValueError("资金、保证金和风控上限须为有限正数")
    if margin_init > 1 or margin_maint > margin_init or commission_per_contract < 0:
        raise ValueError("保证金比例或手续费无效")
    if max_market_age_seconds < 1:
        raise ValueError("行情最大时效须为正整数")
    config = BlackSectorConfig(
        leader_products=tuple(item.strip().upper() for item in leader_products),
        comparison_product=comparison_product.strip().upper(),
        execution_product=execution_product.strip().upper(),
        signal_role=signal_role.strip().lower(),
        execution_role=execution_role.strip().lower(),
        target_key=f"{execution_product.strip().lower()}_{execution_role.strip().lower()}",
        quantity=quantity, return_period=return_period, sector_period=sector_period,
        submission_delay_bars=submission_delay_bars,
    )
    loaded = load_sector_research(
        bars_dir=bars_dir, contract_struct_path=contract_struct,
        signal_products=config.signal_products, signal_role=config.signal_role,
        execution_product=config.execution_product, execution_role=config.execution_role,
        end_day=end_day,
    )
    print(f"研究数据加载耗时={perf_counter() - started_at:.1f}s", flush=True)
    ends = dict(loaded.day_end_ns)
    days = tuple(day for day in sorted(ends) if start_day <= day <= end_day)
    if not days:
        raise ValueError(f"请求区间 {start_day}..{end_day} 没有可用 Bar")
    long_gaps = tuple((left, right) for left, right in zip(days, days[1:])
                      if right - left > MAX_COVERAGE_GAP)
    print(f"行情覆盖: 请求={start_day}..{end_day} 实际={days[0]}..{days[-1]} "
          f"交易日数={len(days)} Bar目录={bars_dir.resolve()}", flush=True)
    if days[0] - start_day > MAX_COVERAGE_GAP or end_day - days[-1] > MAX_COVERAGE_GAP or long_gaps:
        raise ValueError(f"行情未覆盖请求区间；超过14天的缺口={long_gaps}，"
                         f"请检查 KLINE_DIR 下的文件")
    assignments = tuple(loaded.store.snapshot(ends[day]) for day in days)
    basic = pd.read_feather(fut_basic)
    product_venues = _validate_basic(basic, config.signal_products)
    needed_by_day: list[tuple[date, dict[str, str]]] = []
    symbol_products: dict[str, str] = {}
    for index, (day, assignment) in enumerate(zip(days, assignments)):
        needed = {assignment.instrument(product, config.signal_role).lower(): product
                  for product in config.signal_products}
        execution_symbol = assignment.instrument(config.execution_product, config.execution_role).lower()
        needed[execution_symbol] = config.execution_product
        if index:
            old_symbol = assignments[index - 1].instrument(
                config.execution_product, config.execution_role,
            ).lower()
            if old_symbol != execution_symbol:
                needed[old_symbol] = config.execution_product
        for symbol, product in needed.items():
            previous = symbol_products.setdefault(symbol, product)
            if previous != product:
                raise ValueError(f"真实合约 {symbol} 映射到多个品种")
        needed_by_day.append((day, needed))
    paths = _bar_paths(bars_dir, {(day, symbol)
                                   for day, needed in needed_by_day for symbol in needed})
    backend = NautilusSimExecutionBackend(
        "black-sector-sim", BacktestEngineConfig(
            trader_id=TraderId("BLACK-SECTOR-001"),
            logging=LoggingConfig(log_level=log_level), run_analysis=True,
        ),
    )
    profiles = {}
    for venue_name in sorted(set(product_venues.values())):
        profile = CtpFuturesBasicProfile(
            profile_id=f"black-sector-{venue_name.lower()}",
            starting_balance=starting_balance,
            commission_per_contract=commission_per_contract,
            venue=Venue(venue_name),
        )
        backend.add_profile(profile)
        profiles[venue_name] = profile
    feed = FileReplayFeed("BLACK_SECTOR_REPLAY")
    instruments = {}
    multipliers = {}
    for symbol, product in sorted(symbol_products.items()):
        instrument, meta, multiplier = _instrument(
            basic, profiles[product_venues[product]], product, symbol,
            margin_init, margin_maint,
        )
        instruments[symbol] = instrument.id
        multipliers[instrument.id] = multiplier
        backend.add_instrument(instrument)
        feed.register_instrument(meta)
    for day, needed in needed_by_day:
        for symbol in sorted(needed):
            path = paths[(day, symbol)]
            feed.add_bar_feather(path, FixedInstrumentBarParser(
                instruments[symbol], timestamp=_timestamp_column(path),
            ))
    resolver = ScheduledContractResolver(tuple(
        ContractAssignment(
            config.target_key,
            instruments[row.instrument(config.execution_product, config.execution_role).lower()],
            row.effective_ns, row.available_ns, revision,
        ) for revision, row in enumerate(assignments, 1)
    ))
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    client = SimulationExecutionClient(
        backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id, positions, prices,
            instrument_limits={item: RiskLimits(
                max_order_quantity=config.quantity * 2,
                max_abs_position=config.quantity,
                max_order_notional=max_notional,
                max_abs_position_notional=max_notional,
                max_market_age_ns=max_market_age_seconds * 1_000_000_000,
                contract_multiplier=multipliers[item],
            ) for item in multipliers},
        ),
    )
    strategy = BlackSectorTargetStrategy("black-sector", loaded.store, config)
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
    adapter = NautilusMarketFeedAdapter(
        "black-sector-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE") for item in instruments.values()),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime("black-sector-runtime", runner, adapter)
    replay_started = perf_counter()
    print(f"回放装配耗时={replay_started - started_at:.1f}s，开始回放", flush=True)
    try:
        result = runtime.run()
        trader = backend.engine.trader
        orders = trader.generate_orders_report()
        fills = trader.generate_fills_report()
        targets = sum(item["kind"] == "target_submitted" for item in strategy.execution_events)
        print(f"days={len(days)} bars={result.replay_summary.bars} "
              f"complete_frames={strategy.complete_frames} signals={len(strategy.signal_events)} "
              f"targets={targets} orders={len(orders)} fills={len(fills)} "
              f"unavailable={strategy.unavailable_events}")
        if not strategy.complete_frames:
            raise AssertionError("没有完整的多品种同步 Bar")
        if client.report_errors:
            raise AssertionError(f"模拟执行回报异常: {client.report_errors}")
        if require_fills and fills.empty:
            raise AssertionError("本次要求成交，但没有模拟成交")
        _report(root=report_dir, backend=backend, result=result.backend_result,
                strategy=strategy, config=config, days=days,
                requested=(start_day, end_day), venues=set(profiles),
                orders=orders, fills=fills, tearsheet=tearsheet)
    finally:
        print(f"历史回放耗时={perf_counter() - replay_started:.1f}s", flush=True)
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="CTP 黑色系次主力信号、主力执行回测")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day", required=True, type=date.fromisoformat)
    parser.add_argument("--leader-products", default="JM,I", help="两个领头品种，逗号分隔")
    parser.add_argument("--comparison-product", default="RB")
    parser.add_argument("--execution-product", default="RB")
    parser.add_argument("--signal-role", default="secondary")
    parser.add_argument("--execution-role", default="main")
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--return-period", type=int, default=30)
    parser.add_argument("--sector-period", type=int, default=15)
    parser.add_argument("--submission-delay-bars", type=int, default=0)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 KLINE_DIR")
    parser.add_argument("--contract-struct", type=Path, help="覆盖 ROLE_DIR 中的合约角色表")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 ROLE_DIR 中的合约基础表")
    parser.add_argument("--starting-balance", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--commission-per-contract", type=Decimal, default=Decimal(1))
    parser.add_argument("--margin-init", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--margin-maint", type=Decimal, default=Decimal("0.08"))
    parser.add_argument("--max-notional", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--max-market-age-seconds", type=int, default=120)
    parser.add_argument("--log-level", choices=("ERROR", "WARNING", "INFO", "DEBUG"), default="WARNING")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--no-tearsheet", action="store_true")
    parser.add_argument("--require-fills", action="store_true")
    args = parser.parse_args()
    role_dir = Path(os.environ["ROLE_DIR"]).expanduser() if os.environ.get("ROLE_DIR") else None
    bars_dir = args.bars_dir or (Path(os.environ["KLINE_DIR"]).expanduser()
                                 if os.environ.get("KLINE_DIR") else None)
    contract_struct = args.contract_struct or (role_dir / "fut_contract_data.feather" if role_dir else None)
    fut_basic = args.fut_basic or (role_dir / "fut_basic.feather" if role_dir else None)
    if bars_dir is None or contract_struct is None or fut_basic is None:
        parser.error("请设置 ROLE_DIR、KLINE_DIR，或分别传入 --bars-dir、--contract-struct、--fut-basic")
    run_case(start_day=args.start_day, end_day=args.end_day,
             bars_dir=bars_dir, contract_struct=contract_struct, fut_basic=fut_basic,
             leader_products=tuple(item.strip() for item in args.leader_products.split(",")),
             comparison_product=args.comparison_product,
             execution_product=args.execution_product,
             signal_role=args.signal_role, execution_role=args.execution_role,
             quantity=args.quantity, return_period=args.return_period,
             sector_period=args.sector_period,
             submission_delay_bars=args.submission_delay_bars,
             starting_balance=args.starting_balance,
             commission_per_contract=args.commission_per_contract,
             margin_init=args.margin_init, margin_maint=args.margin_maint,
             max_notional=args.max_notional,
             max_market_age_seconds=args.max_market_age_seconds,
             log_level=args.log_level, report_dir=args.report_dir,
             tearsheet=not args.no_tearsheet, require_fills=args.require_fills)


if __name__ == "__main__":
    main()
