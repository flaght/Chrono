"""CTP 品种动态主力 EMA 的离线 Bar 回测入口。只连接模拟交易端。"""

from __future__ import annotations

import argparse
import json
import pandas as pd
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from time import perf_counter


from dotenv import load_dotenv 
# 读取本地环境配置；命令行显式路径仍由公共解析器优先采用。
load_dotenv()



from bomber.framework.dataprep.scenarios.role_futures import prepare_role_research as load_sector_research
from bomber.framework.dataprep.bars import add_bar_source
from bomber.framework.dataprep.contracts import BarReadSpec
from bomber.framework.dataprep.paths import resolve_futures_args
from bomber.framework.dataprep.session import input_session, read_feather, write_input_reports



from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from bomber.framework.market.basic.base import DataType
from bomber.framework.market.replay.base import FileReplayFeed

from bomber.framework.trader import (
    ContractAssignment, CtpFuturesBasicProfile, DataBinding,
    DynamicExecutionRoute, MarketReferencePriceStore, MarketStreamBinding,
    NautilusMarketFeedAdapter, NautilusSimExecutionBackend,
    NetTargetOrderPlanner, PositionManager, PreTradeRiskManager, RiskLimits,
    RuntimeMode, ScheduledContractResolver, SimulationExecutionClient,
    UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)
from bomber.framework.dataprep.catalog import bar_path as _bar_path
from bomber.framework.dataprep.futures import instrument as _instrument
from bomber.framework.dataprep.metadata import venue as _venue_for_product


from strategy import MainEmaConfig, MainEmaStrategy

DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"
MAX_COVERAGE_GAP = timedelta(days=14)


def _start_phase(label: str) -> float:
    """用单调时钟记录阶段墙钟时间，立即输出以定位长时间等待的阶段。"""
    print(f"[耗时] 开始：{label}", flush=True)
    return perf_counter()


def _finish_phase(label: str, started_at: float) -> None:
    print(f"[耗时] {label}完成，耗时={perf_counter() - started_at:.3f}s", flush=True)



def _write_reports(*, report_dir: Path, backend, backend_result,
                   strategy: MainEmaStrategy, days: tuple[date, ...],
                   product: str, venue: Venue,
                   orders: pd.DataFrame, fills: pd.DataFrame,
                   positions: pd.DataFrame, tearsheet: bool) -> None:
    """把可核对的原始报表与绩效指标保存到本次回测目录。"""
    report_dir = report_dir / str(backend_result.run_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    # 输入依据和交易结果保存在同一个运行目录，必须在输入会话退出前写出。
    write_input_reports(report_dir)
    trader = backend.engine.trader
    reports = {
        "orders.csv": orders,
        "fills.csv": fills,
        "positions.csv": positions,
        "account.csv": trader.generate_account_report(venue),
    }
    for filename, frame in reports.items():
        frame.to_csv(report_dir / filename)
    summary = asdict(backend_result) if is_dataclass(backend_result) else {
        "backend_result": str(backend_result),
    }
    summary["strategy"] = {
        "product": product,
        "venue": str(venue),
        "start_day": days[0].isoformat(),
        "end_day": days[-1].isoformat(),
        "ema_bars": strategy.bars_used,
        "last_main": strategy.last_main,
        "last_target": None if strategy.last_target is None else str(strategy.last_target),
        "unavailable_events": strategy.unavailable_events,
        "orders_report_rows": len(orders),
        "fills_report_rows": len(fills),
        "positions_report_rows": len(positions),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"回测报表: {report_dir.resolve()}")
    if tearsheet:
        try:
            # 按需加载绘图依赖；--no-tearsheet 只输出 JSON/CSV。
            from bomber.analysis.tearsheet import create_tearsheet

            chart_path = report_dir / "tearsheet.html"
            create_tearsheet(
                engine=backend.engine,
                output_path=str(chart_path),
                title=f"{product} Main EMA Backtest",
            )
        except ImportError as exc:
            print(f"绩效图未生成：缺少可视化依赖（{exc}）；CSV 和 summary.json 已保存")
        else:
            print(f"交互绩效图: {chart_path.resolve()}")


# main 与 run_case 嵌套时复用同一会话，避免原始文件重复读取。
@input_session
def run_case(*, product: str, start_day: date, end_day: date, bars_dir: Path,
             contract_struct: Path, fut_basic: Path, factors_path: Path,
             factor_availability: str = "aligned",
             fast: int, slow: int,
             quantity: Decimal, require_fills: bool = False,
             log_level: str = "WARNING",
             report_dir: Path = DEFAULT_REPORT_DIR,
             tearsheet: bool = True,
             max_notional: Decimal = Decimal("1000000"), bar_timestamp: str = "end") -> None:
    """准备一个品种的主力输入，装配历史运行器，执行模拟回测并输出报表。"""
    case_started_at = perf_counter()
    product = product.strip().upper()
    if not product.isalpha():
        raise ValueError("--product 须为品种字母代码，例如 RB、I、HC")
    if end_day < start_day:
        raise ValueError("日期范围无效")
    if not max_notional.is_finite() or max_notional <= 0:
        raise ValueError("最大名义金额须为正且有限")
        
    phase_started_at = _start_phase("角色、累计因子和研究行情加载")
    # main 同时用于信号和执行。因子按当前交易日匹配，symbol 冲突时因子表优先。
    # 公共层仍扫描截至 end_day 的历史准备范围，耗时不只取决于回测区间长度。
    loaded = load_sector_research(
        bar_timestamp=bar_timestamp,
        bars_dir=bars_dir, contract_struct_path=contract_struct,
        products=(product,), signal_role="main",
        end_day=end_day,
        execution_product=product, execution_role="main",
        factors_path=factors_path, factor_date_basis="trading",
        factor_availability=factor_availability,
    )
    _finish_phase("角色、累计因子和研究行情加载", phase_started_at)
    phase_started_at = _start_phase("日期覆盖检查和主力合约选择")
    ends = dict(loaded.day_end_ns)
    # 研究准备范围可更大；实际送入 EMA 的分钟行情仅来自请求区间。
    days = tuple(day for day in sorted(ends) if start_day <= day <= end_day)
    if not days:
        raise ValueError(
            f"{product} 在 {start_day} 至 {end_day} 没有可用 Bar；"
            f"Bar目录={bars_dir.resolve()}，已识别行情范围="
            f"{min(ends, default=None)} 至 {max(ends, default=None)}",
        )
    print(
        f"{product} 行情覆盖: Bar目录={bars_dir.resolve()} "
        f"请求={start_day}..{end_day} 实际={days[0]}..{days[-1]} 交易日数={len(days)}",
        flush=True,
    )
    long_gaps = tuple((previous, current) for previous, current in zip(days, days[1:])
                      if current - previous > MAX_COVERAGE_GAP)
    if (days[0] - start_day > MAX_COVERAGE_GAP
            or end_day - days[-1] > MAX_COVERAGE_GAP or long_gaps):
        raise ValueError(
            f"{product} 行情未覆盖请求区间：实际只到 {days[-1]}，"
            f"超过14天的缺口={long_gaps}；请检查 KLINE_DIR 下的文件路径、"
            "文件名和数据日期后重跑",
        )
    # 日级快照用于预装合约和路由；策略运行时仍按每根 Bar 的事件时间查询。
    assignments = tuple(loaded.store.snapshot(ends[day]) for day in days)
    symbols = {row.instrument(product, "main") for row in assignments}
    for previous, current in zip(assignments, assignments[1:]):
        if previous.instrument(product, "main") != current.instrument(product, "main"):
            symbols.add(previous.instrument(product, "main"))
    _finish_phase("日期覆盖检查和主力合约选择", phase_started_at)
    phase_started_at = _start_phase("基础条款加载和校验")
    basic = read_feather(fut_basic)
    # 真实合约 tick、乘数、交易所和生命周期均从资料读取，不写死在策略中。
    required = {"code", "symbol", "exchangeCD", "contMultNum", "minChgPriceNum",
                "listDate", "lastTradeDate"}
    if required - set(basic.columns):
        raise ValueError(f"fut_basic 缺少列: {sorted(required - set(basic.columns))}")
    venue = Venue(_venue_for_product(basic, product))
    config = MainEmaConfig(
        product=product, venue=str(venue), fast_period=fast,
        slow_period=slow, quantity=quantity,
    )
    _finish_phase("基础条款加载和校验", phase_started_at)

    phase_started_at = _start_phase("模拟后端初始化和真实合约注册")
    backend = NautilusSimExecutionBackend(
        f"{product.lower()}-main-ema-sim",
        BacktestEngineConfig(
            trader_id=TraderId(f"{product}-MAIN-EMA-001"),
            logging=LoggingConfig(log_level=log_level),
            run_analysis=True,
        ),
    )
    profile = CtpFuturesBasicProfile(
        # 示例模拟账户：固定起始资金和每手手续费，不是完整 CTP 结算模型。
        starting_balance=Decimal("1000000"),
        commission_per_contract=Decimal(1), venue=venue,
    )
    backend.add_profile(profile)
    feed = FileReplayFeed(f"{product}_MAIN_EMA_REPLAY")
    instruments = {}
    multipliers = {}
    for symbol in sorted(symbols):
        # 原生合约交给撮合后端，元信息交给 Feed，乘数用于名义金额风控。
        instrument, meta, multiplier = _instrument(basic, profile, product, symbol)
        instruments[symbol] = instrument.id
        multipliers[instrument.id] = multiplier
        backend.add_instrument(instrument)
        feed.register_instrument(meta)
    _finish_phase("模拟后端初始化和真实合约注册", phase_started_at)

    phase_started_at = _start_phase("执行行情标准化和注册")
    for index, (day, row) in enumerate(zip(days, assignments)):
        needed = {row.instrument(product, "main")}
        # 换月时旧仓仍需真实报价才能处理，因此当日必须同时加载旧主力行情。
        if index and assignments[index - 1].instrument(product, "main") != row.instrument(product, "main"):
            needed.add(assignments[index - 1].instrument(product, "main"))
        for symbol in sorted(needed):
            path = _bar_path(bars_dir, symbol, day)
            # 标准化 start/end 分钟标签并严格校验 OHLCV，注册准备好的真实行。
            add_bar_source(feed, path, instruments[symbol],
                spec=BarReadSpec(timestamp_label=bar_timestamp))
    _finish_phase("执行行情标准化和注册", phase_started_at)

    phase_started_at = _start_phase("路由、风控和运行器装配")
    # 将 rb_main 这类逻辑目标按生效/可用时间映射到最终真实主力合约。
    resolver = ScheduledContractResolver(tuple(
        ContractAssignment(config.target_key, instruments[row.instrument(product, "main")],
                           row.effective_ns, row.available_ns, revision)
        for revision, row in enumerate(assignments, 1)
    ))
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    # 仓位差规划器生成订单，风控按真实报价、合约乘数和行情年龄检查。
    client = SimulationExecutionClient(
        backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id, positions, prices,
            instrument_limits={item: RiskLimits(
                max_order_quantity=2 * quantity, max_abs_position=quantity,
                max_order_notional=max_notional,
                max_abs_position_notional=max_notional,
                max_market_age_ns=120 * 1_000_000_000,
                contract_multiplier=multipliers[item],
            ) for item in multipliers},
        ),
    )
    strategy = MainEmaStrategy(f"{product.lower()}-main-ema", loaded.store, config)
    adapter = NautilusMarketFeedAdapter(
        # 适配器将同一真实行情连接到策略回放与模拟后端，生命周期由 runtime 管理。
        f"{product.lower()}-main-ema-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE")
              for item in instruments.values()), manage_lifecycle=False,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    # 报价观察器维护风控参考价；绑定决定策略收哪些 Bar，路由决定目标发往哪里。
    runner.add_market_observer(prices)
    runner.add_data_feed("main-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(DataBinding(str(item), "main-bars", item, DataType.BAR, "1-MINUTE")
                            for item in instruments.values()),
        execution_routes=(DynamicExecutionRoute(config.target_key, backend.backend_id, resolver),),
    )
    runtime = UnifiedHistoricalRuntime(f"{product.lower()}-main-ema-runtime", runner, adapter)
    _finish_phase("路由、风控和运行器装配", phase_started_at)
    print(f"[耗时] 数据准备及回放装配累计={perf_counter() - case_started_at:.3f}s", flush=True)
    try:
        phase_started_at = _start_phase("历史回测运行")
        # 该阶段含行情回放、策略信号、模拟执行及运行器内部分析。
        result = runtime.run()
        _finish_phase("历史回测运行", phase_started_at)
        phase_started_at = _start_phase("交易报表生成和结果检查")
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        positions_report = backend.engine.trader.generate_positions_report()
        print(f"product={product} venue={venue} days={len(days)} bars={result.replay_summary.bars} "
              f"ema_bars={strategy.bars_used} main={strategy.last_main} "
              f"target={strategy.last_target} orders={len(orders)} fills={len(fills)} "
              f"unavailable={strategy.unavailable_events}")
        if strategy.bars_used < config.slow_period:
            raise AssertionError("有效主力 Bar 不足以完成 EMA 预热")
        if client.report_errors:
            raise AssertionError(f"模拟执行回报异常: {client.report_errors}")
        if require_fills and fills.empty:
            raise AssertionError("本次要求验收成交，但没有模拟成交")
        _finish_phase("交易报表生成和结果检查", phase_started_at)
        phase_started_at = _start_phase("报表写出及可选绩效图生成")
        _write_reports(
            report_dir=report_dir, backend=backend,
            backend_result=result.backend_result, strategy=strategy, days=days,
            product=product, venue=venue,
            orders=orders, fills=fills, positions=positions_report,
            tearsheet=tearsheet,
        )
        _finish_phase("报表写出及可选绩效图生成", phase_started_at)
    finally:
        # 回测或报表出错也停止运行器，不吞掉原异常。
        phase_started_at = _start_phase("运行器停止")
        runtime.stop()
        _finish_phase("运行器停止", phase_started_at)
    print(f"[耗时] run_case 总耗时={perf_counter() - case_started_at:.3f}s", flush=True)


@input_session
def main() -> None:
    parser = argparse.ArgumentParser(description="CTP 品种动态主力 EMA 离线回测")
    parser.add_argument("--product", required=True, help="品种代码，例如 RB、I、HC")
    parser.add_argument("--start-day", type=date.fromisoformat, required=True)
    parser.add_argument("--end-day", type=date.fromisoformat, required=True)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 KLINE_DIR")
    parser.add_argument("--data-root", type=Path, help="CTP 数据根目录")
    parser.add_argument("--bar-timestamp", choices=("start", "end"), default="end", help="源分钟标签；公共层统一为结束时刻")
    parser.add_argument("--contract-struct", type=Path, help="覆盖 ROLE_DIR 中的合约角色表")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 ROLE_DIR 中的合约基础表")
    parser.add_argument("--factors", type=Path, help="外部累计复权因子；默认角色表同目录 fut_adjustment_factors.feather")
    parser.add_argument("--factor-availability", choices=("aligned", "explicit"), default="aligned",
                        help="默认使用上游已对齐当日因子；explicit 强制要求 available_ns")
    parser.add_argument("--fast", type=int, default=3)
    parser.add_argument("--slow", type=int, default=5)
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--require-fills", action="store_true")
    parser.add_argument(
        "--log-level", choices=("ERROR", "WARNING", "INFO", "DEBUG"),
        default="WARNING", help="框架日志级别；默认仅显示警告和错误",
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--no-tearsheet", action="store_true", help="只导出 CSV 和 JSON")
    parser.add_argument("--max-notional", type=Decimal, default=Decimal("1000000"),
                        help="模拟风控的单笔与持仓名义金额上限")
    args = parser.parse_args()

    main_started_at = perf_counter()
    phase_started_at = _start_phase("输入路径解析")
    try:
        # 这里只解析并检查路径；资料内容在 run_case 中读取。
        paths = resolve_futures_args(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    _finish_phase("输入路径解析", phase_started_at)
    bars_dir = bars = bars_root = paths.fut
    contract_struct = roles = paths.contract_struct
    fut_basic = basic = paths.fut_basic
    run_case(bar_timestamp=args.bar_timestamp, product=args.product, start_day=args.start_day, end_day=args.end_day,
             bars_dir=bars_dir,
             contract_struct=contract_struct,
             fut_basic=fut_basic,
             factors_path=args.factors or contract_struct.parent / "fut_adjustment_factors.feather",
             factor_availability=args.factor_availability,
             fast=args.fast, slow=args.slow, quantity=args.quantity,
             require_fills=args.require_fills, log_level=args.log_level,
             report_dir=args.report_dir, tearsheet=not args.no_tearsheet,
             max_notional=args.max_notional)
    print(f"[耗时] 路径解析至回测报表完成总耗时={perf_counter() - main_started_at:.3f}s", flush=True)


if __name__ == "__main__":
    main()
