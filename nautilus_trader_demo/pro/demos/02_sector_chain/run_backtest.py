"""CTP 可配置产业链回测：主力复权价生成信号，次主力执行。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter
import pandas as pd


from dotenv import load_dotenv 
load_dotenv()

from bomber.framework.dataprep.bars import add_bar_source
from bomber.framework.dataprep.contracts import BarReadSpec
from bomber.framework.dataprep.paths import resolve_futures_args
from bomber.framework.dataprep.session import input_session, read_feather, write_input_reports
from bomber.framework.dataprep.scenarios.role_futures import prepare_role_research

from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from bomber.framework.market.basic.base import DataType
from bomber.framework.market.replay.base import FileReplayFeed
from bomber.framework.trader import (
    ContractAssignment, CtpFuturesBasicProfile, DataBinding, DynamicExecutionRoute,
    MarketReferencePriceStore, MarketStreamBinding, NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend, NetTargetOrderPlanner, PositionManager,
    PreTradeRiskManager, RiskLimits, RuntimeMode, ScheduledContractResolver,
    SimulationExecutionClient, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)

from strategy import SectorChainConfig, SectorChainTargetStrategy

MAX_COVERAGE_GAP = timedelta(days=14)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"


def _start_phase(label: str) -> float:
    print(f"[耗时] 开始：{label}", flush=True)
    return perf_counter()


def _finish_phase(label: str, started_at: float) -> None:
    print(f"[耗时] {label}完成，耗时={perf_counter() - started_at:.3f}s", flush=True)


from bomber.framework.dataprep.catalog import bar_paths as _bar_paths
from bomber.framework.dataprep.futures import instrument as _instrument
from bomber.framework.dataprep.metadata import selected_products as _validate_basic


def _report(*, root: Path, backend, result, strategy: SectorChainTargetStrategy,
            config: SectorChainConfig, days: tuple[date, ...], requested: tuple[date, date],
            venues: set[str], orders: pd.DataFrame, fills: pd.DataFrame,
            tearsheet: bool) -> None:
    output = root / str(result.run_id)
    output.mkdir(parents=True, exist_ok=True)
    write_input_reports(output)
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
                             title="CTP Sector Chain Backtest")
        except ImportError as exc:
            print(f"绩效图未生成（{exc}）；CSV 和 JSON 已保存")
        else:
            print(f"交互绩效图: {chart.resolve()}")


@input_session
def run_case(*, start_day: date, end_day: date, bars_dir: Path,
             contract_struct: Path, fut_basic: Path, factors_path: Path,
             factor_availability: str = "aligned",
             leader_products: tuple[str, str] = ("JM", "I"),
             comparison_product: str = "RB", execution_product: str = "RB",
             signal_role: str = "main", execution_role: str = "secondary",
             quantity: Decimal = Decimal(1), return_period: int = 30,
             sector_period: int = 15, submission_delay_bars: int = 0,
             starting_balance: Decimal = Decimal("1000000"),
             commission_per_contract: Decimal = Decimal(1),
             margin_init: Decimal = Decimal("0.10"), margin_maint: Decimal = Decimal("0.08"),
             max_notional: Decimal = Decimal("1000000"),
             max_market_age_seconds: int = 120,
             log_level: str = "WARNING", report_dir: Path = DEFAULT_REPORT_DIR,
             tearsheet: bool = True, require_fills: bool = False, bar_timestamp: str = "end") -> None:
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
    config = SectorChainConfig(
        leader_products=tuple(item.strip().upper() for item in leader_products),
        comparison_product=comparison_product.strip().upper(),
        execution_product=execution_product.strip().upper(),
        signal_role=signal_role.strip().lower(),
        execution_role=execution_role.strip().lower(),
        target_key=f"{execution_product.strip().lower()}_{execution_role.strip().lower()}",
        quantity=quantity, return_period=return_period, sector_period=sector_period,
        submission_delay_bars=submission_delay_bars,
    )
    phase_started_at = _start_phase("角色、累计因子和研究行情加载")
    loaded = prepare_role_research(
        bar_timestamp=bar_timestamp,
        bars_dir=bars_dir, contract_struct_path=contract_struct,
        products=config.signal_products, signal_role=config.signal_role,
        execution_product=config.execution_product, execution_role=config.execution_role,
        factors_path=factors_path, factor_date_basis="trading",
        factor_availability=factor_availability,
        end_day=end_day,
    )
    _finish_phase("角色、累计因子和研究行情加载", phase_started_at)
    phase_started_at = _start_phase("日期覆盖检查和角色快照选择")
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
                         f"请检查 FUT_KLINE_DATA_DIR 下的文件")
    assignments = tuple(loaded.store.snapshot(ends[day]) for day in days)
    _finish_phase("日期覆盖检查和角色快照选择", phase_started_at)
    phase_started_at = _start_phase("基础条款加载和校验")
    basic = read_feather(fut_basic)
    product_venues = _validate_basic(basic, config.signal_products)
    _finish_phase("基础条款加载和校验", phase_started_at)
    phase_started_at = _start_phase("信号、执行及换约合约文件选择")
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
    _finish_phase("信号、执行及换约合约文件选择", phase_started_at)
    phase_started_at = _start_phase("模拟后端初始化和真实合约注册")
    backend = NautilusSimExecutionBackend(
        "sector-chain-sim", BacktestEngineConfig(
            trader_id=TraderId("SECTOR-CHAIN-001"),
            logging=LoggingConfig(log_level=log_level), run_analysis=True,
        ),
    )
    profiles = {}
    for venue_name in sorted(set(product_venues.values())):
        profile = CtpFuturesBasicProfile(
            profile_id=f"sector-chain-{venue_name.lower()}",
            starting_balance=starting_balance,
            commission_per_contract=commission_per_contract,
            venue=Venue(venue_name),
        )
        backend.add_profile(profile)
        profiles[venue_name] = profile
    feed = FileReplayFeed("SECTOR_CHAIN_REPLAY")
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
    _finish_phase("模拟后端初始化和真实合约注册", phase_started_at)
    phase_started_at = _start_phase("行情标准化和数据源注册")
    for day, needed in needed_by_day:
        for symbol in sorted(needed):
            path = paths[(day, symbol)]
            add_bar_source(feed, path, instruments[symbol],
                spec=BarReadSpec(timestamp_label=bar_timestamp))
    _finish_phase("行情标准化和数据源注册", phase_started_at)
    phase_started_at = _start_phase("路由、风控和运行器装配")
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
    strategy = SectorChainTargetStrategy("sector-chain", loaded.store, config)
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
        "sector-chain-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE") for item in instruments.values()),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime("sector-chain-runtime", runner, adapter)
    _finish_phase("路由、风控和运行器装配", phase_started_at)
    print(f"[耗时] 数据准备及回放装配累计={perf_counter() - started_at:.3f}s", flush=True)
    try:
        phase_started_at = _start_phase("历史回测运行")
        result = runtime.run()
        _finish_phase("历史回测运行", phase_started_at)
        phase_started_at = _start_phase("交易报表生成和结果检查")
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
        _finish_phase("交易报表生成和结果检查", phase_started_at)
        phase_started_at = _start_phase("报表写出及可选绩效图生成")
        _report(root=report_dir, backend=backend, result=result.backend_result,
                strategy=strategy, config=config, days=days,
                requested=(start_day, end_day), venues=set(profiles),
                orders=orders, fills=fills, tearsheet=tearsheet)
        _finish_phase("报表写出及可选绩效图生成", phase_started_at)
    finally:
        phase_started_at = _start_phase("运行器停止")
        runtime.stop()
        _finish_phase("运行器停止", phase_started_at)
    print(f"[耗时] run_case 总耗时={perf_counter() - started_at:.3f}s", flush=True)


@input_session
def main() -> None:
    parser = argparse.ArgumentParser(description="CTP 产业链主力信号、次主力执行回测")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day", required=True, type=date.fromisoformat)
    parser.add_argument("--leader-products", default="JM,I", help="两个领头品种，逗号分隔")
    parser.add_argument("--comparison-product", default="RB", help="比较品种，不限产业链")
    parser.add_argument("--execution-product", help="执行品种，默认比较品种，须属于三个信号品种")
    parser.add_argument("--signal-role", default="main", help="信号角色，默认主力")
    parser.add_argument("--execution-role", default="secondary", help="执行角色，默认次主力")
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--return-period", type=int, default=30)
    parser.add_argument("--sector-period", type=int, default=15)
    parser.add_argument("--submission-delay-bars", type=int, default=0)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 FUT_KLINE_DATA_DIR")
    parser.add_argument("--data-root", type=Path, help="CTP 数据根目录")
    parser.add_argument("--bar-timestamp", choices=("start", "end"), default="end", help="源分钟标签；公共层统一为结束时刻")
    parser.add_argument("--contract-struct", type=Path, help="覆盖 FUT_ROLE_DATA_DIR 中的合约角色表")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 FUT_ROLE_DATA_DIR 中的合约基础表")
    parser.add_argument("--factors", type=Path, help="外部累计复权因子；默认角色表同目录 fut_adjustment_factors.feather")
    parser.add_argument("--factor-availability", choices=("aligned", "explicit"), default="aligned",
                        help="默认使用上游已对齐当日因子；explicit 强制要求 available_ns")
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
    main_started_at = perf_counter()
    phase_started_at = _start_phase("输入路径解析")
    try:
        paths = resolve_futures_args(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    _finish_phase("输入路径解析", phase_started_at)
    bars_dir = bars = bars_root = paths.fut
    contract_struct = roles = paths.contract_struct
    fut_basic = basic = paths.fut_basic
    run_case(bar_timestamp=args.bar_timestamp, start_day=args.start_day, end_day=args.end_day,
             bars_dir=bars_dir, contract_struct=contract_struct, fut_basic=fut_basic,
             factors_path=args.factors or contract_struct.parent / "fut_adjustment_factors.feather",
             factor_availability=args.factor_availability,
             leader_products=tuple(item.strip() for item in args.leader_products.split(",")),
             comparison_product=args.comparison_product,
             execution_product=args.execution_product or args.comparison_product,
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
    print(f"[耗时] 路径解析至回测报表完成总耗时={perf_counter() - main_started_at:.3f}s", flush=True)


if __name__ == "__main__":
    main()
