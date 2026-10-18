"""使用真实期货合约行情的多品种主力横截面动量回测入口。"""

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
    # 直接启动脚本时，将项目根目录加入模块搜索路径。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from bomber.framework.dataprep.bars import add_prepared_bar_source
from bomber.framework.dataprep.contracts import BarFileKey, BarReadSpec
from bomber.framework.dataprep.paths import resolve_futures_args
from bomber.framework.dataprep.session import input_session, read_feather, write_input_reports

from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId
from bomber.framework.dataprep.scenarios.role_futures import prepare_role_research
from bomber.framework.dataprep.scenarios.fixed_contracts import plan_fixed_contracts, prepare_fixed_contracts
from strategy import MainCrossSectionMomentumStrategy
from bomber.framework.market.basic.base import DataType
from bomber.framework.market.replay.base import FileReplayFeed
from bomber.framework.trader import (
    CtpFuturesBasicProfile, DataBinding, ExecutionRoute, MarketReferencePriceStore,
    MarketStreamBinding, NautilusMarketFeedAdapter, NautilusSimExecutionBackend,
    NetTargetOrderPlanner, PositionManager, PreTradeRiskManager, RiskLimits,
    RuntimeMode, SimulationExecutionClient, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)



from bomber.framework.dataprep.futures import instrument as _instrument
from bomber.framework.dataprep.metadata import selected_products as _selected_products

DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"


def _start_phase(name: str) -> float:
    """打印阶段开始信息，并记录单调计时器的起点。"""
    print(f"[耗时] 开始：{name}", flush=True)
    return perf_counter()


def _finish_phase(name: str, started: float) -> None:
    """打印该阶段的实际耗时，不受系统时钟调整影响。"""
    print(f"[耗时] {name}完成，耗时={perf_counter() - started:.3f}s", flush=True)


@input_session
def run_case(*, products: tuple[str, ...], start_day: date, end_day: date,
             bars_dir: Path, contract_struct: Path, fut_basic: Path, lookback: int = 20,
             factors_path: Path | None = None, factor_availability: str = "aligned",
             rebalance_interval: int = 5, target_notional: Decimal = Decimal("100000"),
             group_fraction: Decimal = Decimal("0.30"),
             starting_balance: Decimal = Decimal("1000000"),
             commission_per_contract: Decimal = Decimal("1"),
             margin_init: Decimal = Decimal("0.10"),
             margin_maint: Decimal = Decimal("0.08"),
             max_notional: Decimal = Decimal("1000000"),
             log_level: str = "WARNING", report_dir: Path | None = None,
             tearsheet: bool = True, require_fills: bool = False, bar_timestamp: str = "end") -> None:
    """准备输入、装配模拟执行、运行回测并输出交易与输入报表。"""
    started = perf_counter()
    if end_day < start_day:
        raise ValueError("日期范围无效")
    if any(not value.is_finite() or value <= 0 for value in
           (target_notional, starting_balance, max_notional, margin_init, margin_maint)) or not commission_per_contract.is_finite() or commission_per_contract < 0:
        raise ValueError("资金、名义金额和风控上限须为正，手续费不能为负")
    if margin_init > 1 or margin_maint > margin_init:
        raise ValueError("保证金比例须满足 0 < margin_maint <= margin_init <= 1")
    products = tuple(item.strip().upper() for item in products)
    if len(products) < 2 or len(set(products)) != len(products) or any(not item.isalpha() for item in products):
        raise ValueError("--products 须为至少两个不重复的品种字母代码")
    if lookback < 1 or rebalance_interval < 1:
        raise ValueError("回看窗口和调仓间隔必须为正整数")
    if not group_fraction.is_finite() or not 0 < group_fraction <= Decimal("0.5"):
        raise ValueError("--group-fraction 须满足 0 < 比例 <= 0.5 且有限")
    phase = _start_phase("合约基础资料加载")
    basic = read_feather(fut_basic)
    venues = _selected_products(basic, products)
    _finish_phase("合约基础资料加载", phase)
    phase = _start_phase("角色、累计因子和研究行情加载")
    # 使用已对齐当前交易日的主力累计因子；同角色合约代码冲突时以因子表为准。
    loaded = prepare_role_research(
        bar_timestamp=bar_timestamp,
        bars_dir=bars_dir, contract_struct_path=contract_struct,
        products=products, signal_role="main",
        end_day=end_day,
        execution_product=products[0], execution_role="main",
        factors_path=factors_path if factors_path is not None else contract_struct.parent / "fut_adjustment_factors.feather",
        factor_date_basis="trading", factor_availability=factor_availability,
    )
    _finish_phase("角色、累计因子和研究行情加载", phase)
    phase = _start_phase("日期覆盖与真实合约输入计划")
    ends = dict(loaded.day_end_ns)
    days = tuple(day for day in sorted(ends) if start_day <= day <= end_day)
    if not days:
        raise ValueError("请求区间没有可用的主力 Bar")
    if days[0] - start_day > timedelta(days=14) or end_day - days[-1] > timedelta(days=14) or any(
        right - left > timedelta(days=14) for left, right in zip(days, days[1:])
    ):
        raise ValueError(f"行情未覆盖请求区间 {start_day}..{end_day}；实际仅 {days[0]}..{days[-1]}")
    assignments = tuple(loaded.store.snapshot(ends[day]) for day in days)
    # 汇总区间内的真实主力及换约旧主力，供合约注册和执行路由使用。
    selected: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(assignments):
        for product in products:
            selected[row.instrument(product, "main").lower()] = (product, venues[product])
            if index and assignments[index - 1].instrument(product, "main") != row.instrument(product, "main"):
                selected[assignments[index - 1].instrument(product, "main").lower()] = (product, venues[product])
    required_bars = set()
    for index, (day, row) in enumerate(zip(days, assignments)):
        for product in products:
            symbols = {row.instrument(product, "main").lower()}
            if index and assignments[index - 1].instrument(product, "main") != row.instrument(product, "main"):
                symbols.add(assignments[index - 1].instrument(product, "main").lower())
            for symbol in symbols:
                # 换约日旧主力也是必需输入，供目标归零订单使用真实行情。
                required_bars.add(BarFileKey("future", symbol, day))
    print(f"products={','.join(products)} requested={start_day}..{end_day} "
          f"actual={days[0]}..{days[-1]} days={len(days)} "
          f"contracts={len(selected)}", flush=True)
    plan = plan_fixed_contracts(required_bars, requested=(start_day, end_day))
    _finish_phase("日期覆盖与真实合约输入计划", phase)
    phase = _start_phase("真实主力与换约行情加载校验")
    bundle = prepare_fixed_contracts(
        bars_dir, plan, spec=BarReadSpec(timestamp_label=bar_timestamp),
    )
    _finish_phase("真实主力与换约行情加载校验", phase)
    phase = _start_phase("模拟账户与真实合约注册")
    backend = NautilusSimExecutionBackend(
        "cross-section-ctp-sim", BacktestEngineConfig(
            trader_id=TraderId("CROSS-SECTION-001"),
            logging=LoggingConfig(log_level=log_level), run_analysis=True,
        ),
    )
    feed = FileReplayFeed("CROSS_SECTION_REPLAY")
    profiles = {}
    instruments = {}
    multipliers = {}
    for symbol, (product, venue_name) in selected.items():
        if venue_name not in profiles:
            # 不同交易所分别建立模拟账户，同一交易所的品种共享账户。
            profile = CtpFuturesBasicProfile(
                profile_id=f"cross-section-{venue_name.lower()}",
                starting_balance=starting_balance,
                commission_per_contract=commission_per_contract,
                venue=Venue(venue_name),
            )
            backend.add_profile(profile)
            profiles[venue_name] = profile
        instrument, meta, multiplier = _instrument(
            basic, profiles[venue_name], product, symbol, margin_init, margin_maint,
        )
        backend.add_instrument(instrument)
        feed.register_instrument(meta)
        instruments[symbol] = instrument.id
        multipliers[instrument.id] = multiplier
    _finish_phase("模拟账户与真实合约注册", phase)
    phase = _start_phase("已准备行情注册到 Feed")
    for source in bundle.sources:
        # 直接注册公共层已校验的行情，避免再次读取原始文件。
        add_prepared_bar_source(feed, source.result, instruments[source.result.key.symbol.lower()])
    _finish_phase("已准备行情注册到 Feed", phase)
    phase = _start_phase("策略、风控与运行器装配")
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    client = SimulationExecutionClient(
        backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id, positions, prices,
            instrument_limits={item: RiskLimits(
                max_order_notional=max_notional,
                max_abs_position_notional=max_notional,
                max_market_age_ns=120 * 1_000_000_000,
                contract_multiplier=multipliers[item],
            ) for item in instruments.values()},
        ),
    )
    strategy = MainCrossSectionMomentumStrategy(
        "cross-section-ctp", products=products, roles=loaded.store,
        instruments=instruments, multipliers=multipliers,
        lookback=lookback, rebalance_interval=rebalance_interval,
        target_notional=target_notional, group_fraction=group_fraction,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    # 先更新真实参考价，再让策略处理行情并提交目标。
    runner.add_market_observer(prices)
    runner.add_data_feed("cross-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(DataBinding(str(item), "cross-bars", item, DataType.BAR, "1-MINUTE")
                            for item in instruments.values()),
        execution_routes=tuple(ExecutionRoute(str(item), backend.backend_id, item)
                               for item in instruments.values()),
    )
    adapter = NautilusMarketFeedAdapter(
        "cross-section-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE") for item in instruments.values()),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime("cross-section-runtime", runner, adapter)
    _finish_phase("策略、风控与运行器装配", phase)
    try:
        phase = _start_phase("运行回测")
        result = runtime.run()
        _finish_phase("运行回测", phase)
        phase = _start_phase("交易报表生成与结果检查")
        trader = backend.engine.trader
        orders = trader.generate_orders_report()
        fills = trader.generate_fills_report()
        print(f"bars={result.replay_summary.bars} frames={strategy.synchronized_frames} "
              f"rebalances={strategy.rebalances} group_size={strategy.signal.group_size} "
              f"orders={len(orders)} fills={len(fills)}")
        if not strategy.rebalances:
            raise AssertionError("共同分钟 Bar 不足以完成回看窗口和调仓")
        if client.report_errors:
            raise AssertionError(f"模拟执行回报异常: {client.report_errors}")
        if require_fills and fills.empty:
            raise AssertionError("本次要求成交，但没有模拟成交")
        positions_report = trader.generate_positions_report()
        accounts = {venue_name: trader.generate_account_report(Venue(venue_name)) for venue_name in profiles}
        _finish_phase("交易报表生成与结果检查", phase)
        phase = _start_phase("报表与绩效图写出")
        output = (report_dir or DEFAULT_REPORT_DIR) / str(result.backend_result.run_id)
        output.mkdir(parents=True, exist_ok=True)
        write_input_reports(output)
        orders.to_csv(output / "orders.csv")
        fills.to_csv(output / "fills.csv")
        positions_report.to_csv(output / "positions.csv")
        for venue_name, account in accounts.items():
            account.to_csv(output / f"account_{venue_name}.csv")
        summary = (asdict(result.backend_result) if is_dataclass(result.backend_result)
                   else {"backend_result": str(result.backend_result)})
        summary["strategy"] = {
            "products": products, "contracts": tuple(selected),
            "lookback": lookback, "rebalance_interval": rebalance_interval,
            "target_notional": str(target_notional), "factor_date_basis": "trading",
            "target_notional_basis": "per_side", "group_fraction": str(group_fraction),
            "group_size": strategy.signal.group_size,
            "last_long_products": strategy.last_signal.long_products if strategy.last_signal else (),
            "last_short_products": strategy.last_signal.short_products if strategy.last_signal else (),
            "factor_availability": factor_availability,
            "requested": [str(start_day), str(end_day)], "actual": [str(days[0]), str(days[-1])],
            "frames": strategy.synchronized_frames, "rebalances": strategy.rebalances,
            "last_targets": {key: str(value) for key, value in (strategy.last_targets or {}).items()},
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                         default=str) + "\n", encoding="utf-8")
        print(f"回测报表: {output.resolve()}")
        if tearsheet:
            try:
                from bomber.analysis.tearsheet import create_tearsheet
                chart = output / "tearsheet.html"
                create_tearsheet(engine=backend.engine, output_path=str(chart),
                                 title="CTP Cross Section Backtest")
                print(f"交互绩效图: {chart.resolve()}")
            except ImportError as exc:
                print(f"绩效图未生成（{exc}）；CSV 和 JSON 已保存")
        _finish_phase("报表与绩效图写出", phase)
    finally:
        # 回测或报表阶段失败时，也尝试关闭运行器。
        phase = _start_phase("运行器清理")
        runtime.stop()
        _finish_phase("运行器清理", phase)
        print(f"[耗时] run_case 总耗时={perf_counter() - started:.3f}s", flush=True)


@input_session
def main() -> None:
    """解析命令行参数与公共数据路径，启动一次横截面回测。"""
    parser = argparse.ArgumentParser(description="CTP 多品种动态主力截面动量回测")
    parser.add_argument("--products", required=True, help="逗号分隔的品种，例如 RB,HC,I")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day", required=True, type=date.fromisoformat)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 FUT_KLINE_DATA_DIR")
    parser.add_argument("--data-root", type=Path, help="CTP 数据根目录")
    parser.add_argument("--bar-timestamp", choices=("start", "end"), default="end", help="源分钟标签；公共层统一为结束时刻")
    parser.add_argument("--contract-struct", type=Path, help="覆盖角色表路径")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 FUT_ROLE_DATA_DIR/fut_basic.feather")
    parser.add_argument("--factors", type=Path, help="默认角色表同目录 fut_adjustment_factors.feather")
    parser.add_argument("--factor-availability", choices=("aligned", "explicit"), default="aligned",
                        help="默认上游已对齐当日因子；explicit 要求 available_ns")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--target-notional", type=Decimal, default=Decimal("100000"),
                        help="每侧总名义金额预算，组内等额分配；不足一手不建仓")
    parser.add_argument("--group-fraction", type=Decimal, default=Decimal("0.30"),
                        help="排名前后每侧选仓比例，默认0.30，上限0.5")
    parser.add_argument("--starting-balance", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--commission-per-contract", type=Decimal, default=Decimal("1"))
    parser.add_argument("--margin-init", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--margin-maint", type=Decimal, default=Decimal("0.08"))
    parser.add_argument("--max-notional", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--log-level", choices=("ERROR", "WARNING", "INFO", "DEBUG"), default="WARNING")
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
    run_case(bar_timestamp=args.bar_timestamp, products=tuple(item.strip() for item in args.products.split(",")),
             start_day=args.start_day, end_day=args.end_day,
             bars_dir=paths.fut, contract_struct=paths.contract_struct, fut_basic=paths.fut_basic,
             factors_path=args.factors, factor_availability=args.factor_availability, lookback=args.lookback,
             rebalance_interval=args.rebalance_interval, target_notional=args.target_notional,
             group_fraction=args.group_fraction,
             starting_balance=args.starting_balance,
             commission_per_contract=args.commission_per_contract,
             margin_init=args.margin_init, margin_maint=args.margin_maint,
             max_notional=args.max_notional, log_level=args.log_level,
             report_dir=args.report_dir, tearsheet=not args.no_tearsheet,
             require_fills=args.require_fills)
    print(f"[耗时] main 总耗时={perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
