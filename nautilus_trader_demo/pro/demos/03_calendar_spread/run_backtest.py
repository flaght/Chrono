"""可指定品种的两期限真实合约价差模拟回测。"""

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

from dotenv import load_dotenv

load_dotenv()

from bomber.framework.dataprep.bars import add_prepared_bar_source
from bomber.framework.dataprep.catalog import product_inventory
from bomber.framework.dataprep.contracts import BarFileKey, BarReadSpec
from bomber.framework.dataprep.paths import resolve_futures_args
from bomber.framework.dataprep.session import input_session, read_feather, write_input_reports

from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from strategy import (
    CalendarSelection, CalendarSpreadConfig, CalendarSpreadStrategy,
)
from bomber.framework.dataprep.futures import instrument
from bomber.framework.dataprep.metadata import venue
from calendar_signal import ROLE_ORDER, select_roles
from bomber.framework.dataprep.scenarios.fixed_contracts import plan_fixed_contracts, prepare_fixed_contracts
from bomber.framework.dataprep.references import load_role_assignments
from bomber.framework.market.basic.base import DataType
from bomber.framework.market.replay.base import FileReplayFeed
from bomber.framework.trader import (
    CtpFuturesBasicProfile, DataBinding, ExecutionRoute, MarketReferencePriceStore,
    MarketStreamBinding, NautilusMarketFeedAdapter, NautilusSimExecutionBackend,
    NetTargetOrderPlanner, PositionManager, PreTradeRiskManager, RiskLimits,
    RuntimeMode, SimulationExecutionClient, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)

MAX_COVERAGE_GAP = timedelta(days=14)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"


def _start_phase(name: str) -> float:
    print(f"[耗时] 开始：{name}", flush=True)
    return perf_counter()


def _finish_phase(name: str, started: float) -> None:
    print(f"[耗时] {name}完成，耗时={perf_counter() - started:.3f}s", flush=True)


@input_session
def run_case(*, product: str, start_day: date, end_day: date, bars_dir: Path,
             contract_struct: Path, fut_basic: Path,
             leg1_role: str = "secondary", leg2_role: str = "far",
             quantity: Decimal = Decimal(1), lookback: int = 120,
             entry_z: float = 2.0, exit_z: float = 0.5,
             rebalance_interval: int = 5,
             starting_balance: Decimal = Decimal("1000000"),
             commission_per_contract: Decimal = Decimal(1),
             margin_init: Decimal = Decimal("0.10"),
             margin_maint: Decimal = Decimal("0.08"),
             max_notional: Decimal = Decimal("1000000"),
             max_market_age_seconds: int = 120,
             missing_role_policy: str = "raise",
             log_level: str = "WARNING", report_dir: Path = DEFAULT_REPORT_DIR,
             tearsheet: bool = True, require_fills: bool = False, bar_timestamp: str = "end") -> None:
    started = perf_counter()
    product = product.strip().upper()
    if not product.isalpha() or end_day < start_day:
        raise ValueError("品种代码或日期范围无效")
    if missing_role_policy not in {"raise", "next-available"}:
        raise ValueError("missing_role_policy 必须为 raise 或 next-available")
    config = CalendarSpreadConfig(
        leg1_role=leg1_role, leg2_role=leg2_role,
        quantity=quantity, lookback=lookback,
        entry_z=entry_z, exit_z=exit_z,
        rebalance_interval=rebalance_interval,
    )
    if any(not value.is_finite() or value <= 0 for value in
           (starting_balance, margin_init, margin_maint, max_notional)):
        raise ValueError("资金、保证金和风控上限须为有限正数")
    if (not commission_per_contract.is_finite() or commission_per_contract < 0
            or margin_init > 1 or margin_maint > margin_init or max_market_age_seconds < 1):
        raise ValueError("手续费、保证金比例或行情最大时效无效")
    phase = _start_phase("行情目录扫描与日期覆盖")
    available = product_inventory(bars_dir, product, start_day, end_day)
    days = tuple(sorted(available))
    if not days:
        raise ValueError(f"{start_day}..{end_day} 没有 {product} Bar")
    gaps = tuple((left, right) for left, right in zip(days, days[1:])
                 if right - left > MAX_COVERAGE_GAP)
    print(f"行情覆盖: product={product} 请求={start_day}..{end_day} 实际={days[0]}..{days[-1]} "
          f"交易日数={len(days)}", flush=True)
    if days[0] - start_day > MAX_COVERAGE_GAP or end_day - days[-1] > MAX_COVERAGE_GAP or gaps:
        raise ValueError(f"行情未覆盖请求区间；超过14天的缺口={gaps}")
    _finish_phase("行情目录扫描与日期覆盖", phase)
    phase = _start_phase("角色资料加载与双腿选择")
    chosen, role_fallbacks = select_roles(
        load_role_assignments(contract_struct, (product,),
            tuple(ROLE_ORDER) if missing_role_policy == "next-available" else (config.leg1_role, config.leg2_role)),
        days, product, leg1_role=config.leg1_role, leg2_role=config.leg2_role,
        available=available, missing_role_policy=missing_role_policy,
    )
    if role_fallbacks:
        print(f"角色表合约缺 Bar，已使用当日有 Bar 的相邻角色（{len(role_fallbacks)} 天）："
              f"{role_fallbacks[:10]}", flush=True)
    _finish_phase("角色资料加载与双腿选择", phase)
    phase = _start_phase("真实双腿及换约行情加载校验")
    required_bars = set()
    optional_old_bars = set()
    previous_pairs = []
    for index, (day, leg1, leg2) in enumerate(chosen):
        required_bars.update({(day, leg1), (day, leg2)})
        old_pair = None
        if index and (chosen[index - 1][1], chosen[index - 1][2]) != (leg1, leg2):
            old_pair = chosen[index - 1][1:]
            optional_old_bars.update((day, symbol) for symbol in old_pair)
        previous_pairs.append(old_pair)
    # 策略声明真实双腿，公共层统一校验并缓存行情；旧合约行情用于换约清仓。
    plan = plan_fixed_contracts(
        (BarFileKey("future", symbol, day) for day, symbol in required_bars),
        (BarFileKey("future", symbol, day) for day, symbol in optional_old_bars),
        requested=(start_day, end_day),
    )
    bundle = prepare_fixed_contracts(
        bars_dir, plan, spec=BarReadSpec(timestamp_label=bar_timestamp),
    )
    prepared = {(source.result.key.trading_day, source.result.key.symbol.lower()): source.result
                for source in bundle.sources}
    paths = {key: result.path for key, result in prepared.items()}
    needed_by_day = tuple((day, {symbol for candidate_day, symbol in paths if candidate_day == day})
                          for day, _, _ in chosen)
    selections = tuple(
        CalendarSelection(
            # 日程先启用，信号仍等待两腿同时间戳。
            day, min(prepared[(day, leg1)].first_ns, prepared[(day, leg2)].first_ns),
            leg1, leg2,
            old_pair_bars_available=(old_pair is None or all(
                (day, symbol) in paths for symbol in old_pair
            )),
        )
        for (day, leg1, leg2), old_pair in zip(chosen, previous_pairs)
    )
    missing_old = tuple((day, symbol) for (day, _, _), old_pair in zip(chosen, previous_pairs)
                        if old_pair for symbol in old_pair if (day, symbol) not in paths)
    if missing_old:
        print(f"换月旧合约缺 Bar（空仓时可跳过）: {missing_old[:10]}", flush=True)
    _finish_phase("真实双腿及换约行情加载校验", phase)
    phase = _start_phase("合约基础资料加载")
    basic = read_feather(fut_basic)
    # 字段、交易所和合约身份校验使用公共 metadata / futures 能力。
    venue_name = venue(basic, product)
    _finish_phase("合约基础资料加载", phase)
    phase = _start_phase("模拟账户与真实合约注册")
    backend = NautilusSimExecutionBackend(
        f"{product.lower()}-calendar-spread-sim", BacktestEngineConfig(
            trader_id=TraderId(f"{product}-CALENDAR-SPREAD-001"),
            logging=LoggingConfig(log_level=log_level), run_analysis=True,
        ),
    )
    profile = CtpFuturesBasicProfile(
        profile_id=f"{product.lower()}-calendar-{venue_name.lower()}",
        starting_balance=starting_balance,
        commission_per_contract=commission_per_contract,
        venue=Venue(venue_name),
    )
    backend.add_profile(profile)
    feed = FileReplayFeed(f"{product}_CALENDAR_REPLAY")
    instruments = {}
    multipliers = {}
    for symbol in sorted({item for _, needed in needed_by_day for item in needed}):
        contract, meta, multiplier = instrument(
            basic, profile, product, symbol, margin_init, margin_maint,
        )
        instruments[symbol] = contract.id
        multipliers[contract.id] = multiplier
        backend.add_instrument(contract)
        feed.register_instrument(meta)
    _finish_phase("模拟账户与真实合约注册", phase)
    phase = _start_phase("已准备行情注册到 Feed")
    for day, needed in needed_by_day:
        for symbol in sorted(needed):
            add_prepared_bar_source(feed, prepared[(day, symbol)], instruments[symbol])
    _finish_phase("已准备行情注册到 Feed", phase)
    phase = _start_phase("策略、风控与运行器装配")
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
            ) for item in instruments.values()},
        ),
    )
    strategy = CalendarSpreadStrategy(
        f"{product.lower()}-calendar-spread", config, selections, instruments,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    # 真实参考价先更新，随后策略在同分钟双腿上提交目标。
    runner.add_market_observer(prices)
    runner.add_data_feed("calendar-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(DataBinding(str(item), "calendar-bars", item, DataType.BAR, "1-MINUTE")
                            for item in instruments.values()),
        execution_routes=tuple(ExecutionRoute(str(item), backend.backend_id, item)
                               for item in instruments.values()),
    )
    adapter = NautilusMarketFeedAdapter(
        f"{product.lower()}-calendar-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE")
              for item in instruments.values()), manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime(f"{product.lower()}-calendar-runtime", runner, adapter)
    _finish_phase("策略、风控与运行器装配", phase)
    try:
        phase = _start_phase("运行回测")
        result = runtime.run()
        _finish_phase("运行回测", phase)
        phase = _start_phase("交易报表生成与结果检查")
        trader = backend.engine.trader
        orders = trader.generate_orders_report()
        fills = trader.generate_fills_report()
        print(f"product={product} days={len(days)} bars={result.replay_summary.bars} "
              f"frames={strategy.complete_frames} submissions={strategy.submissions} "
              f"rolls={strategy.rolls} z={strategy.last_z} "
              f"orders={len(orders)} fills={len(fills)}")
        if strategy._old_pair is not None:
            raise AssertionError(f"换约未完成：旧合约对 {strategy._old_pair} 未安全清仓")
        if strategy.complete_frames <= config.lookback:
            raise AssertionError("双期限同分钟 Bar 不足以完成窗口预热")
        if client.report_errors:
            raise AssertionError(f"模拟执行回报异常: {client.report_errors}")
        if require_fills and fills.empty:
            raise AssertionError("本次要求成交，但没有模拟成交")
        positions_report = trader.generate_positions_report()
        account_report = trader.generate_account_report(Venue(venue_name))
        _finish_phase("交易报表生成与结果检查", phase)
        phase = _start_phase("报表与绩效图写出")
        output = report_dir / str(result.backend_result.run_id)
        output.mkdir(parents=True, exist_ok=True)
        write_input_reports(output)
        orders.to_csv(output / "orders.csv")
        fills.to_csv(output / "fills.csv")
        positions_report.to_csv(output / "positions.csv")
        account_report.to_csv(output / "account.csv")
        summary = (asdict(result.backend_result) if is_dataclass(result.backend_result)
                   else {"backend_result": str(result.backend_result)})
        summary["strategy"] = {
            "product": product, "venue": venue_name,
            "missing_role_policy": missing_role_policy,
            "role_fallbacks": role_fallbacks,
            "leg1_role": config.leg1_role, "leg2_role": config.leg2_role,
            "requested": [str(start_day), str(end_day)],
            "actual": [str(days[0]), str(days[-1])],
            "frames": strategy.complete_frames,
            "submissions": strategy.submissions, "rolls": strategy.rolls,
            "direction": strategy.direction, "last_z": strategy.last_z,
            "last_targets": {key: str(value) for key, value in (strategy.last_targets or {}).items()},
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
                                 title=f"{product} Calendar Spread Backtest")
            except ImportError as exc:
                print(f"绩效图未生成（{exc}）；CSV 和 JSON 已保存")
            else:
                print(f"交互绩效图: {chart.resolve()}")
        _finish_phase("报表与绩效图写出", phase)
    finally:
        phase = _start_phase("运行器清理")
        runtime.stop()
        _finish_phase("运行器清理", phase)
        print(f"[耗时] run_case 总耗时={perf_counter() - started:.3f}s", flush=True)


@input_session
def main() -> None:
    parser = argparse.ArgumentParser(description="单品种两期限价差均值回归回测")
    parser.add_argument("--product", required=True, help="品种，例如 RB、HC、I、JM")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day", required=True, type=date.fromisoformat)
    parser.add_argument("--leg1-role", "--near-role", dest="leg1_role",
                        choices=("main", "secondary"), default="secondary",
                        help="第一腿角色，默认次主力 secondary；near-role 仅为旧参数别名")
    parser.add_argument("--leg2-role", "--far-role", dest="leg2_role",
                        choices=("secondary", "far"), default="far",
                        help="第二腿角色，默认远期 far")
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 FUT_KLINE_DATA_DIR")
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
    parser.add_argument("--missing-role-policy", choices=("raise", "next-available"),
                        default="raise")
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
    bars_dir = paths.fut
    contract_struct = paths.contract_struct
    fut_basic = paths.fut_basic
    run_case(bar_timestamp=args.bar_timestamp, product=args.product, start_day=args.start_day, end_day=args.end_day,
             bars_dir=bars_dir, contract_struct=contract_struct,
             fut_basic=fut_basic, leg1_role=args.leg1_role, leg2_role=args.leg2_role,
             quantity=args.quantity, lookback=args.lookback,
             entry_z=args.entry_z, exit_z=args.exit_z,
             rebalance_interval=args.rebalance_interval,
             starting_balance=args.starting_balance,
             commission_per_contract=args.commission_per_contract,
             margin_init=args.margin_init, margin_maint=args.margin_maint,
             max_notional=args.max_notional,
             max_market_age_seconds=args.max_market_age_seconds,
             missing_role_policy=args.missing_role_policy,
             log_level=args.log_level, report_dir=args.report_dir,
             tearsheet=not args.no_tearsheet, require_fills=args.require_fills)
    print(f"[耗时] main 总耗时={perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
