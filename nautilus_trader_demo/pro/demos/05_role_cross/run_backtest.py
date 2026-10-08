"""CTP 单品种主力、次主力、远期复权价穿越的可配置历史回测。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bomber.framework.dataprep.bars import add_prepared_bar_source
from bomber.framework.dataprep.contracts import BarFileKey, BarReadSpec
from bomber.framework.dataprep.paths import resolve_futures_args
from bomber.framework.dataprep.session import input_session, read_feather, write_input_reports

import pandas as pd

from dotenv import load_dotenv 
load_dotenv()

from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from bomber.framework.datahub import MinimalDataHub
from bomber.framework.dataprep.scenarios.role_futures import prepare_role_research
from bomber.framework.dataprep.scenarios.fixed_contracts import plan_fixed_contracts, prepare_fixed_contracts
from role_signal import SIGNAL_ROLES
from strategy import RoleCrossConfig, RoleCrossTargetStrategy
from bomber.framework.market.basic.base import DataType
from bomber.framework.market.replay.base import FileReplayFeed
from bomber.framework.trader import (
    ContractAssignment, CtpFuturesBasicProfile, DataBinding, DynamicExecutionRoute,
    MarketReferencePriceStore, MarketStreamBinding, NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend, NetTargetOrderPlanner, PositionManager,
    PreTradeRiskManager, RiskLimits, RuntimeMode, ScheduledContractResolver,
    SimulationExecutionClient, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)

MAX_COVERAGE_GAP = timedelta(days=14)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"


from bomber.framework.dataprep.futures import instrument as _instrument
from bomber.framework.dataprep.metadata import venue as _venue


def _start_phase(name: str) -> float:
    """打印阶段名称并记录计时起点。"""
    print(f"[耗时] 开始：{name}", flush=True)
    return perf_counter()


def _finish_phase(name: str, started: float) -> None:
    """打印独立阶段耗时，便于区分数据准备、回测和报表成本。"""
    print(f"[耗时] {name}完成，耗时={perf_counter() - started:.3f}s", flush=True)


def _report(*, root: Path, backend, result, strategy: RoleCrossTargetStrategy,
            product: str, venue: str, days: tuple[date, ...],
            requested: tuple[date, date], orders: pd.DataFrame,
            fills: pd.DataFrame, factor_gaps: tuple,
            factor_anchors: tuple, tearsheet: bool) -> None:
    """写出交易、输入和角色换约因子诊断，可选生成绩效图。"""
    output = root / str(result.run_id)
    output.mkdir(parents=True, exist_ok=True)
    write_input_reports(output)
    trader = backend.engine.trader
    orders.to_csv(output / "orders.csv")
    fills.to_csv(output / "fills.csv")
    trader.generate_positions_report().to_csv(output / "positions.csv")
    trader.generate_account_report(Venue(venue)).to_csv(output / "account.csv")
    summary = asdict(result) if is_dataclass(result) else {"backend_result": str(result)}
    summary["strategy"] = {
        "product": product, "venue": venue,
        "requested": [str(value) for value in requested],
        "actual": [str(days[0]), str(days[-1])],
        "complete_frames": strategy.complete_frames,
        "signals": len(strategy.signal_events),
        "unavailable_events": strategy.unavailable_events,
        "factor_gaps": factor_gaps,
        "factor_anchors": factor_anchors,
        "orders": len(orders), "fills": len(fills),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"回测报表: {output.resolve()}")
    if tearsheet:
        try:
            from bomber.analysis.tearsheet import create_tearsheet
            chart = output / "tearsheet.html"
            create_tearsheet(engine=backend.engine, output_path=str(chart),
                             title=f"{product} Three-Role Cross Backtest")
        except ImportError as exc:
            print(f"绩效图未生成（{exc}）；CSV 和 JSON 已保存")
        else:
            print(f"交互绩效图: {chart.resolve()}")


@input_session
def run_case(*, product: str, start_day: date, end_day: date,
             bars_dir: Path, contract_struct: Path, fut_basic: Path,
             quantity: Decimal = Decimal(1),
             starting_balance: Decimal = Decimal("1000000"),
             commission_per_contract: Decimal = Decimal(1),
             margin_init: Decimal = Decimal("0.10"),
             margin_maint: Decimal = Decimal("0.08"),
             max_notional: Decimal = Decimal("1000000"),
             max_market_age_seconds: int = 120,
             log_level: str = "WARNING", report_dir: Path = DEFAULT_REPORT_DIR,
             tearsheet: bool = True, require_fills: bool = False, bar_timestamp: str = "end") -> None:
    """准备三角色研究价与真实行情，装配主力动态执行并运行回测。"""
    started = perf_counter()
    product = product.strip().upper()
    if not product.isalpha() or end_day < start_day:
        raise ValueError("品种代码或日期范围无效")
    if any(not value.is_finite() or value <= 0 for value in
           (quantity, starting_balance, margin_init, margin_maint, max_notional)):
        raise ValueError("目标手数、资金、保证金和风控上限须为有限正数")
    if quantity != quantity.to_integral_value():
        raise ValueError("期货目标手数必须为正整数")
    if (not commission_per_contract.is_finite() or commission_per_contract < 0
            or margin_init > 1 or margin_maint > margin_init or max_market_age_seconds < 1):
        raise ValueError("手续费、保证金比例或行情最大时效无效")
    phase = _start_phase("三角色研究行情与换约因子加载")
    # 分钟角色场景使用各角色自己的因果换约因子，不套用主力外部因子。
    loaded = prepare_role_research(
        bar_timestamp=bar_timestamp,
        bars_dir=bars_dir, contract_struct_path=contract_struct,
        products=(product,), minute_roles=SIGNAL_ROLES, end_day=end_day,
    )
    _finish_phase("三角色研究行情与换约因子加载", phase)
    print(f"研究数据 files={loaded.file_count} bars={loaded.bar_count}", flush=True)
    phase = _start_phase("日期覆盖与真实合约输入计划")
    ends = dict(loaded.day_end_ns)
    days = tuple(day for day in sorted(ends) if start_day <= day <= end_day)
    if not days:
        raise ValueError(f"{product} 在 {start_day}..{end_day} 没有可用 Bar")
    gaps = tuple((left, right) for left, right in zip(days, days[1:])
                 if right - left > MAX_COVERAGE_GAP)
    print(f"行情覆盖: product={product} 请求={start_day}..{end_day} "
          f"实际={days[0]}..{days[-1]} 交易日数={len(days)}", flush=True)
    if days[0] - start_day > MAX_COVERAGE_GAP or end_day - days[-1] > MAX_COVERAGE_GAP or gaps:
        raise ValueError(f"行情未覆盖请求区间；超过14天的缺口={gaps}")
    assignments = tuple(loaded.store.assignment_at(ends[day]) for day in days)
    needed_by_day = []
    symbols = set()
    for index, (day, assignment) in enumerate(zip(days, assignments)):
        needed = set(assignment.contracts.values())
        if index and assignments[index - 1].contracts["main"] != assignment.contracts["main"]:
            needed.add(assignments[index - 1].contracts["main"])
        symbols.update(needed)
        needed_by_day.append((day, needed))
    # 三角色当日行情必需；主力变化时还需要旧主力行情来处理旧仓。
    plan = plan_fixed_contracts(
        (BarFileKey("future", symbol, day) for day, needed in needed_by_day for symbol in needed),
        requested=(start_day, end_day),
    )
    _finish_phase("日期覆盖与真实合约输入计划", phase)
    phase = _start_phase("真实三角色及换约行情加载校验")
    bundle = prepare_fixed_contracts(
        bars_dir, plan, spec=BarReadSpec(timestamp_label=bar_timestamp),
    )
    _finish_phase("真实三角色及换约行情加载校验", phase)
    phase = _start_phase("合约基础资料加载")
    basic = read_feather(fut_basic)
    # 基础条款字段、交易所和合约身份由公共元数据模块校验。
    venue = _venue(basic, product)
    config = RoleCrossConfig(target_key=f"{product.lower()}_main",
                             venue=venue, quantity=quantity)
    _finish_phase("合约基础资料加载", phase)
    phase = _start_phase("模拟账户与真实合约注册")
    backend = NautilusSimExecutionBackend(
        f"{product.lower()}-role-cross-sim", BacktestEngineConfig(
            trader_id=TraderId(f"{product}-ROLE-CROSS-001"),
            logging=LoggingConfig(log_level=log_level), run_analysis=True,
        ),
    )
    profile = CtpFuturesBasicProfile(
        profile_id=f"role-cross-{venue.lower()}",
        starting_balance=starting_balance,
        commission_per_contract=commission_per_contract,
        venue=Venue(venue),
    )
    backend.add_profile(profile)
    feed = FileReplayFeed(f"{product}_ROLE_CROSS_REPLAY")
    instruments = {}
    multipliers = {}
    for symbol in sorted(symbols):
        instrument, meta, multiplier = _instrument(
            basic, profile, product, symbol, margin_init, margin_maint,
        )
        instruments[symbol] = instrument.id
        multipliers[instrument.id] = multiplier
        backend.add_instrument(instrument)
        feed.register_instrument(meta)
    _finish_phase("模拟账户与真实合约注册", phase)
    phase = _start_phase("已准备行情注册到回放源")
    for source in bundle.sources:
        # 公共文件键采用大写，研究角色映射采用小写，在入口统一匹配。
        add_prepared_bar_source(feed, source.result, instruments[source.result.key.symbol.lower()])
    _finish_phase("已准备行情注册到回放源", phase)
    phase = _start_phase("主力动态路由、风控与运行器装配")
    resolver = ScheduledContractResolver(tuple(
        ContractAssignment(
            config.target_key, instruments[row.contracts["main"]],
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
    strategy = RoleCrossTargetStrategy(
        f"{product.lower()}-role-cross", MinimalDataHub(loaded.store), config,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    # 先更新真实合约参考价格，再交给策略处理同分钟行情。
    runner.add_market_observer(prices)
    runner.add_data_feed("role-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(DataBinding(str(item), "role-bars", item, DataType.BAR, "1-MINUTE")
                            for item in instruments.values()),
        execution_routes=(DynamicExecutionRoute(config.target_key, backend.backend_id, resolver),),
    )
    adapter = NautilusMarketFeedAdapter(
        f"{product.lower()}-role-cross-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE")
              for item in instruments.values()),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime(f"{product.lower()}-role-cross-runtime", runner, adapter)
    _finish_phase("主力动态路由、风控与运行器装配", phase)
    try:
        phase = _start_phase("运行回测")
        result = runtime.run()
        _finish_phase("运行回测", phase)
        phase = _start_phase("订单成交报表与结果检查")
        trader = backend.engine.trader
        orders = trader.generate_orders_report()
        fills = trader.generate_fills_report()
        print(f"product={product} days={len(days)} bars={result.replay_summary.bars} "
              f"complete_frames={strategy.complete_frames} "
              f"signals={len(strategy.signal_events)} orders={len(orders)} "
              f"fills={len(fills)} factor_gaps={len(loaded.store.factor_gaps)} "
              f"unavailable={strategy.unavailable_events}")
        if not strategy.complete_frames:
            raise AssertionError("没有主力、次主力、远期同分钟完整 Bar")
        if client.report_errors:
            raise AssertionError(f"模拟执行回报异常: {client.report_errors}")
        if require_fills and fills.empty:
            raise AssertionError("本次要求成交，但没有模拟成交")
        _finish_phase("订单成交报表与结果检查", phase)
        phase = _start_phase("完整报表生成与写出")
        _report(root=report_dir, backend=backend, result=result.backend_result,
                strategy=strategy, product=product, venue=venue,
                days=days, requested=(start_day, end_day),
                orders=orders, fills=fills,
                factor_gaps=loaded.store.factor_gaps,
                factor_anchors=loaded.store.factor_anchors,
                tearsheet=tearsheet)
        _finish_phase("完整报表生成与写出", phase)
    finally:
        phase = _start_phase("运行器清理")
        runtime.stop()
        _finish_phase("运行器清理", phase)
        print(f"[耗时] run_case 总耗时={perf_counter() - started:.3f}s", flush=True)


@input_session
def main() -> None:
    """解析策略参数和公共输入路径，启动单品种回测。"""
    parser = argparse.ArgumentParser(description="CTP 品种主力、次主力、远期复权价穿越回测")
    parser.add_argument("--product", required=True, help="品种，例如 RB、I、HC")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day", required=True, type=date.fromisoformat)
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--bars-dir", type=Path, help="覆盖 FUT_KLINE_DATA_DIR 中的合约Bar目录")
    parser.add_argument("--data-root", type=Path, help="CTP 数据根目录")
    parser.add_argument("--bar-timestamp", choices=("start", "end"), default="end", help="源分钟标签；公共层统一为结束时刻")
    parser.add_argument("--contract-struct", type=Path, help="覆盖 FUT_ROLE_DATA_DIR 中的角色表")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 FUT_ROLE_DATA_DIR 中的合约基础表")
    parser.add_argument("--starting-balance", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--commission-per-contract", type=Decimal, default=Decimal(1))
    parser.add_argument("--margin-init", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--margin-maint", type=Decimal, default=Decimal("0.08"))
    parser.add_argument("--max-notional", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--max-market-age-seconds", type=int, default=120)
    parser.add_argument("--log-level", choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        default="WARNING")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--no-tearsheet", action="store_true")
    parser.add_argument("--require-fills", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    phase = _start_phase("输入路径解析")
    try:
        paths = resolve_futures_args(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    _finish_phase("输入路径解析", phase)
    run_case(bar_timestamp=args.bar_timestamp, product=args.product, start_day=args.start_day, end_day=args.end_day,
             bars_dir=paths.fut, contract_struct=paths.contract_struct, fut_basic=paths.fut_basic,
             quantity=args.quantity, starting_balance=args.starting_balance,
             commission_per_contract=args.commission_per_contract,
             margin_init=args.margin_init, margin_maint=args.margin_maint,
             max_notional=args.max_notional,
             max_market_age_seconds=args.max_market_age_seconds,
             log_level=args.log_level, report_dir=args.report_dir,
             tearsheet=not args.no_tearsheet, require_fills=args.require_fills)
    print(f"[耗时] main 总耗时={perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
