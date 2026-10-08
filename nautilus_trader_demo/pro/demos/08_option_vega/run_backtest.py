"""当前统一策略框架的期权Vega卖方/期货Delta对冲离线入口。"""
from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from vega_signal import Contract, OptionVegaConfig, remaining_trading_days, filter_hedgeable_options
from bomber.framework.dataprep.bars import add_prepared_bar_source
from bomber.framework.dataprep.calendar import infer_market_calendar, load_calendar
from bomber.framework.dataprep.catalog import inventory
from bomber.framework.dataprep.contracts import BarFileKey, BarReadSpec
from bomber.framework.dataprep.metadata import load_cffex_futures, load_options_basic, positive
from bomber.framework.dataprep.paths import resolve_option_args
from bomber.framework.dataprep.scenarios.option_chain import plan_option_chain, prepare_option_chain
from bomber.framework.dataprep.session import input_session, write_input_reports

DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"


def _start_phase(name):
    """打印阶段起点，单独计量数据准备与回测。"""
    print(f"[耗时] 开始：{name}", flush=True)
    return perf_counter()


def _finish_phase(name, started):
    """打印已完成阶段的耗时。"""
    print(f"[耗时] {name}完成，耗时={perf_counter() - started:.3f}s", flush=True)


def load_contracts(opt_path, fut_path, option_product, future_product, index_code):
    """使用公共条款读取器，再组装本策略需要的认购与同月期货关系。"""
    specs = load_options_basic(opt_path, option_product, index_code,
        kinds=("C",), require_month_dates=False, require_currency=True)
    futures = load_cffex_futures(fut_path, future_product, require_execution=True)
    contracts = {key: Contract(key, future_product + spec.month[2:], float(spec.strike),
        spec.multiplier, spec.listed, spec.last_trade, spec.expiry, spec.tick)
        for key, spec in specs.items()}
    if not contracts or not futures:
        raise ValueError("没有目标品种的认购期权或期货基础资料")
    return contracts, futures


@input_session
def run_case(args):
    """准备期权链、同月期货和指数行情，运行卖方与实际持仓对冲回测。"""
    import pandas as pd
    from bomber.backtest.config import BacktestEngineConfig
    from bomber.config import LoggingConfig
    from bomber.model import Venue
    from bomber.model.identifiers import InstrumentId, TraderId
    from bomber.framework.market.basic.base import DataType, InstrumentMeta
    from bomber.framework.market.replay.base import FileReplayFeed
    from bomber.framework.trader import (CtpFuturesBasicProfile, DataBinding, ExecutionRoute,
        MarketReferencePriceStore, MarketStreamBinding, NautilusMarketFeedAdapter,
        NautilusSimExecutionBackend, NetTargetOrderPlanner, PositionManager,
        PreTradeRiskManager, RiskLimits, RuntimeMode, SimulationExecutionClient,
        UnifiedHistoricalRuntime, UnifiedStrategyRunner)
    from bomber.framework.trader.instrument_factory import instrument_meta, make_future, make_option
    from strategy import OptionVegaStrategy

    started = perf_counter()
    phase = _start_phase("输入路径解析与配置校验")
    args = resolve_option_args(args, require_futures=True)
    config = OptionVegaConfig(
        nav=args.nav if args.nav is not None else args.starting_balance, vega_budget=args.vega_budget,
        delta_targets=tuple(float(item.strip()) for item in args.delta_targets.split(",")),
        option_slot=args.option_slot, hedge_slots=tuple(args.hedge_slots.split(",")),
        flatten_slot=args.flatten_slot, min_remaining_days=args.min_remaining_days,
        close_remaining_days=args.close_remaining_days, rate=args.rate,
        min_time_value=args.min_time_value, max_market_age_seconds=args.max_market_age_seconds,
        max_iv_age_seconds=args.max_iv_age_seconds, max_option_lots=args.max_option_lots,
        max_future_lots=args.max_future_lots, max_total_option_lots=args.max_total_option_lots,
    )
    for name in ("starting_balance", "max_notional", "margin_init", "margin_maint",
                 "option_margin_init", "option_margin_maint"):
        positive(getattr(args, name), name)
    if args.margin_init > 1 or args.margin_maint > args.margin_init or args.option_margin_maint > args.option_margin_init:
        raise ValueError("保证金比例无效")
    if not args.commission.is_finite() or args.commission < 0:
        raise ValueError("手续费须为有限非负数")
    _finish_phase("输入路径解析与配置校验", phase)
    phase = _start_phase("交易日历加载或推导")
    if args.calendar is not None:
        calendar = load_calendar(args.calendar, args.start_day, args.end_day)
        calendar_source = str(args.calendar)
    else:
        calendar = infer_market_calendar((args.fut_dir, args.opt_dir, args.index_dir),
                                         args.start_day, args.end_day)
        calendar_source = "MARKET_DATES"
    print(f"交易日来源={calendar_source} 日期范围={calendar[0]}..{calendar[-1]}")
    days = tuple(day for day in calendar if args.start_day <= day <= args.end_day)
    if not days:
        raise ValueError("请求区间没有交易日")
    _finish_phase("交易日历加载或推导", phase)
    phase = _start_phase("期权与期货基础条款加载")
    contracts, future_rows = load_contracts(args.opt_basic, args.fut_basic,
        args.option_product, args.future_product, args.index_code)
    _finish_phase("期权与期货基础条款加载", phase)
    phase = _start_phase("行情目录扫描与同月对冲资格组装")
    opt_paths = inventory(args.opt_dir, args.start_day, args.end_day, lambda key: key in contracts)
    fut_paths = inventory(args.fut_dir, args.start_day, args.end_day, lambda key: key in future_rows)
    idx_paths = inventory(args.index_dir, args.start_day, args.end_day, lambda key: key == args.index_code)
    needed = {}
    options_by_day = {}
    skipped_hedges = []
    for day in days:
        candidates = tuple(sorted(key for candidate_day, key in opt_paths
            if candidate_day == day and contracts[key].list_day <= day <= contracts[key].last_day))
        if not candidates or (day, args.index_code) not in idx_paths:
            raise FileNotFoundError(f"{day}: 缺少{args.option_product}认购链或指数{args.index_code}，不截断回测")
        candidates, skipped = filter_hedgeable_options(candidates, contracts, future_rows, day)
        for key in skipped:
            row = future_rows.get(key)
            reason = ("MISSING_BASIC" if row is None else
                      "NOT_LISTED" if day < row["listDate"] else "EXPIRED")
            skipped_hedges.append({"date": str(day), "future": key, "reason": reason})
        if not candidates:
            raise ValueError(f"{day}: 无可用已上市同月期货对冲的期权候选，跳过的期货={skipped}")
        needed[(day, args.index_code)] = idx_paths[(day, args.index_code)]
        for key in candidates:
            info = contracts[key]
            if info.future not in future_rows:
                raise ValueError(
                    f"{day}/{key}: 同月期货{info.future}基本资料未匹配；"
                    f"实际读取资料={args.fut_basic}，品种={args.future_product}"
                )
            if (day, info.future) not in fut_paths:
                expected = Path(args.fut_dir) / f"{day:%Y%m%d}" / f"{info.future}_{day:%Y%m%d}.feather"
                raise FileNotFoundError(
                    f"{day}/{key}: 同月期货{info.future}基本资料已找到，但未匹配当天行情；"
                    f"实际行情目录={args.fut_dir}，预期文件={expected}；"
                    "文件名合约代码允许大小写差异"
                )
            future = future_rows[info.future]
            # 远期到期在已知日期之后时，只需足够日期判断开仓/清仓阈值。
            remaining_trading_days(calendar, day, info.last_day, config.min_remaining_days)
            needed[(day, key)] = opt_paths[(day, key)]
            needed[(day, info.future)] = fut_paths[(day, info.future)]
        options_by_day[day] = candidates
    contracts = {key: value for key, value in contracts.items() if any(key in items for items in options_by_day.values())}
    future_rows = {key: value for key, value in future_rows.items() if any(info.future == key for info in contracts.values())}
    # 对持有过的月份，后续存在的旧合约行情也加载，保证退出后仍能估值/清仓。
    for (day, key), path in {**fut_paths, **opt_paths}.items():
        if day in days and key in contracts:
            needed[(day, key)] = path
        elif day in days and key in future_rows:
            row = future_rows[key]
            if row["listDate"] <= day <= row["lastTradeDate"]:
                needed[(day, key)] = path

    _finish_phase("行情目录扫描与同月对冲资格组装", phase)
    phase = _start_phase("期权链、期货与指数行情加载校验")
    # 三类资产分别索引，严格准备完成后由回放源消费，不再次读取原文件。
    plan = plan_option_chain((BarFileKey(
        "option" if key in contracts else "index" if key == args.index_code else "future", key, day,
    ) for day, key in needed), requested=(args.start_day, args.end_day))
    spec = BarReadSpec(timestamp_label=args.bar_timestamp, trading_day_policy="day_session")
    bundle = prepare_option_chain(
        {"option": args.opt_dir, "future": args.fut_dir, "index": args.index_dir}, plan,
        specs={kind: spec for kind in ("option", "future", "index")},
    )
    prepared = {(source.result.key.trading_day, source.result.key.symbol): source.result for source in bundle.sources}
    frames = {identity: source.frame for identity, source in prepared.items()}
    _finish_phase("期权链、期货与指数行情加载校验", phase)
    phase = _start_phase("开仓同步与末日清仓时钟检查")
    # 指数提供最终清仓的信号时钟。候选合约可能未交易或已提前平仓，
    # 不应要求每个候选都有收盘行情；实际持仓的新鲜报价由策略逐分钟检查。
    final_cutoff = pd.Timestamp(f"{days[-1]} {config.flatten_slot}", tz="Asia/Shanghai")
    final_index = frames[(days[-1], args.index_code)]
    if final_index.datetime.max() < final_cutoff + pd.Timedelta(minutes=2):
        raise ValueError(f"{days[-1]}/{args.index_code}: 指数在最终清仓时点后不足两分钟行情，调早--flatten-slot")
    for day in days:
        minutes = set(frames[(day, args.index_code)].datetime)
        synchronized = any(pd.Timestamp(f"{day} {config.option_slot}", tz="Asia/Shanghai") in
            minutes & set(frames[(day, key)].datetime) & set(frames[(day, contracts[key].future)].datetime)
            for key in options_by_day[day])
        if not synchronized:
            raise ValueError(f"{day}: --option-slot时点没有三腿同步M1行情")

    if skipped_hedges:
        print(f"已排除不可用的同月对冲期货（日期/合约/原因）: {skipped_hedges}")
    print(f"行情覆盖: 请求={args.start_day}..{args.end_day} 实际={days[0]}..{days[-1]} "
          f"交易日={len(days)} 认购合约={len(contracts)} 同月期货={len(future_rows)}")
    _finish_phase("开仓同步与末日清仓时钟检查", phase)
    phase = _start_phase("模拟账户与真实合约注册")
    backend = NautilusSimExecutionBackend("option-vega-sim", BacktestEngineConfig(
        trader_id=TraderId("OPTION-VEGA-001"), logging=LoggingConfig(log_level=args.log_level), run_analysis=True))
    profile = CtpFuturesBasicProfile(profile_id="option-vega-basic", venue=Venue("CFFEX"),
        starting_balance=args.starting_balance, commission_per_contract=args.commission)
    backend.add_profile(profile)
    feed = FileReplayFeed("OPTION_VEGA_REPLAY")
    instruments = {}
    multipliers = {}
    try:
        for key, info in contracts.items():
            contract = make_option(info, args.option_margin_init, args.option_margin_maint, args.index_code)
            instruments[key] = contract.id
            multipliers[contract.id] = info.multiplier
            backend.add_instrument(contract)
            feed.register_instrument(instrument_meta(contract))
        for key, row in future_rows.items():
            contract = make_future(key, row, args.margin_init, args.margin_maint)
            instruments[key] = contract.id
            multipliers[contract.id] = row["contMultNum"]
            backend.add_instrument(contract)
            feed.register_instrument(instrument_meta(contract))
        # 指数是信号数据，只有Feed订阅，不伪造期货或创建执行路由。
        index_id = InstrumentId.from_str(args.index_code + ".XSHG")
        feed.register_instrument(InstrumentMeta(index_id, price_precision=args.index_precision,
            size_precision=0, price_increment=Decimal(1).scaleb(-args.index_precision), exchange="XSHG"))
        all_ids = {**instruments, args.index_code: index_id}
        _finish_phase("模拟账户与真实合约注册", phase)
        phase = _start_phase("已准备行情注册到回放源")
        for identity in sorted(frames):
            add_prepared_bar_source(feed, prepared[identity], all_ids[identity[1]])
        _finish_phase("已准备行情注册到回放源", phase)
        phase = _start_phase("策略、执行路由与风控装配")
        positions = PositionManager()
        prices = MarketReferencePriceStore()
        client = SimulationExecutionClient(backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
            risk_manager=PreTradeRiskManager(backend.backend_id, positions, prices,
                instrument_limits={item: RiskLimits(
                    max_order_quantity=(config.max_option_lots if key in contracts else config.max_future_lots) * 2,
                    max_abs_position=config.max_option_lots if key in contracts else config.max_future_lots,
                    max_order_notional=args.max_notional, max_abs_position_notional=args.max_notional,
                    max_market_age_ns=config.max_market_age_seconds * 1_000_000_000,
                    contract_multiplier=multipliers[item]) for key, item in instruments.items()}))
        strategy = OptionVegaStrategy("option-vega", config, contracts, future_rows, instruments,
                                      args.index_code, calendar, days[-1], options_by_day)
        runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
        runner.add_market_observer(prices)
        runner.add_data_feed("option-bars", feed)
        runner.add_execution_client(client)
        runner.add_strategy(strategy,
            data_bindings=tuple(DataBinding(key, "option-bars", item, DataType.BAR, "1-MINUTE") for key, item in all_ids.items()),
            execution_routes=tuple(ExecutionRoute(str(item), backend.backend_id, item) for item in instruments.values()))
        adapter = NautilusMarketFeedAdapter("option-vega-clock", feed, backend,
            tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE") for item in instruments.values()), manage_lifecycle=False)
        runtime = UnifiedHistoricalRuntime("option-vega-runtime", runner, adapter)
        _finish_phase("策略、执行路由与风控装配", phase)
        try:
            phase = _start_phase("运行回测")
            result = runtime.run()
            _finish_phase("运行回测", phase)
            phase = _start_phase("交易报表与残留持仓检查资料生成")
            trader = backend.engine.trader
            orders, fills = trader.generate_orders_report(), trader.generate_fills_report()
            position_report = trader.generate_positions_report()
            account_report = trader.generate_account_report(Venue("CFFEX"))
            _finish_phase("交易报表与残留持仓检查资料生成", phase)
            phase = _start_phase("报表写出与最终结果校验")
            output = Path(args.report_dir) / str(result.backend_result.run_id)
            output.mkdir(parents=True, exist_ok=True)
            write_input_reports(output)
            orders.to_csv(output / "orders.csv")
            fills.to_csv(output / "fills.csv")
            position_report.to_csv(output / "positions.csv")
            account_report.to_csv(output / "account.csv")
            pd.DataFrame(strategy.signals).to_csv(output / "signals.csv", index=False)
            remaining = {key: str(strategy.account_position(str(item))) for key, item in instruments.items()
                         if strategy.account_position(str(item)) or strategy.working_quantity(str(item))}
            errors = [str(value) for value in client.report_errors]
            summary = {"requested": [str(args.start_day), str(args.end_day)],
                "actual": [str(days[0]), str(days[-1])], "trading_days": len(days),
                "calendar_source": calendar_source,
                "skipped_unavailable_hedges": skipped_hedges,
                "calendar_range": [str(calendar[0]), str(calendar[-1])],
                "option_product": args.option_product, "future_product": args.future_product,
                "index": args.index_code, "bar_timestamp": args.bar_timestamp,
                "bars": result.replay_summary.bars, "frames": strategy.frames,
                "entry_decisions": strategy.entry_decisions, "submissions": strategy.submissions,
                "orders": len(orders), "fills": len(fills), "remaining_positions_or_working": remaining,
                "report_errors": errors, "config": vars(config),
                "simulation": "NETTING/BAR/FIXED_FEE/PREMIUM_PERCENT_MARGIN; NO_EXPIRY_SETTLEMENT",
                "backend": str(result.backend_result)}
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
            print(f"frames={strategy.frames} entries={strategy.entry_decisions} orders={len(orders)} fills={len(fills)} 报表={output}")
            if remaining or errors or not strategy.flatten_requested:
                raise RuntimeError(f"回测未安全结束：残留仓位/订单={remaining} 回报异常={errors}")
            if args.require_fills and fills.empty:
                raise RuntimeError("要求成交，但没有成交；检查Vega预算、价格和选约条件")
            _finish_phase("报表写出与最终结果校验", phase)
            phase = _start_phase("可选绩效图生成")
            if not args.no_tearsheet:
                try:
                    from bomber.analysis.tearsheet import create_tearsheet
                    create_tearsheet(engine=backend.engine, output_path=str(output / "tearsheet.html"), title="Option Vega Backtest")
                except ImportError as exc:
                    print(f"绩效图依赖不可用: {exc}；CSV/JSON已保存")
            _finish_phase("可选绩效图生成", phase)
        finally:
            phase = _start_phase("运行器清理")
            runtime.stop()
            _finish_phase("运行器清理", phase)
    finally:
        # 配置阶段失败时也释放Backend；正常runtime.stop后不会重复释放。
        if not backend.is_disposed:
            backend.stop()
        print(f"[耗时] run_case 总耗时={perf_counter() - started:.3f}s", flush=True)


@input_session
def main():
    """解析期权组合、数据路径及风险限制后启动回测。"""
    parser = argparse.ArgumentParser(description="认购期权卖方Vega预算与同月期货Delta对冲")
    parser.add_argument("--start-day", type=date.fromisoformat, required=True)
    parser.add_argument("--end-day", type=date.fromisoformat, required=True)
    parser.add_argument("--data-root", type=Path)
    for name in ("fut-dir", "opt-dir", "index-dir", "fut-basic", "opt-basic"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--calendar", type=Path, help="可选交易日历；未配置时由完整行情日期自动推导")
    parser.add_argument("--option-product", default="MO")
    parser.add_argument("--future-product", default="IM")
    parser.add_argument("--index-code", default="000852")
    parser.add_argument("--bar-timestamp", choices=("start", "end"), default="end",
                        help="源分钟标签，默认 end；开始标签数据需显式选择 start")
    parser.add_argument("--index-precision", type=int, default=4)
    parser.add_argument("--starting-balance", type=Decimal, default=Decimal("40000000"))
    parser.add_argument("--nav", type=Decimal)
    parser.add_argument("--vega-budget", type=float, default=1.25)
    parser.add_argument("--delta-targets", default="0.2,0.25,0.3,0.35")
    parser.add_argument("--option-slot", default="13:58")
    parser.add_argument("--hedge-slots", default="10:00,11:00,13:30,14:00,14:55")
    parser.add_argument("--flatten-slot", default="14:50")
    parser.add_argument("--min-remaining-days", type=int, default=5)
    parser.add_argument("--close-remaining-days", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--min-time-value", type=float, default=5)
    parser.add_argument("--max-market-age-seconds", type=int, default=300)
    parser.add_argument("--max-iv-age-seconds", type=int, default=300)
    parser.add_argument("--max-option-lots", type=int, default=100)
    parser.add_argument("--max-total-option-lots", type=int, default=200)
    parser.add_argument("--max-future-lots", type=int, default=100)
    parser.add_argument("--max-notional", type=Decimal, default=Decimal("100000000"))
    parser.add_argument("--commission", type=Decimal, default=Decimal(1))
    parser.add_argument("--margin-init", type=Decimal, default=Decimal("0.15"))
    parser.add_argument("--margin-maint", type=Decimal, default=Decimal("0.12"))
    parser.add_argument("--option-margin-init", type=Decimal, default=Decimal(1))
    parser.add_argument("--option-margin-maint", type=Decimal, default=Decimal(1))
    parser.add_argument("--log-level", choices=("ERROR", "WARNING", "INFO", "DEBUG"), default="WARNING")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--no-tearsheet", action="store_true")
    parser.add_argument("--require-fills", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    args.option_product = args.option_product.strip().upper()
    args.future_product = args.future_product.strip().upper()
    if not args.option_product.isalpha() or not args.future_product.isalpha() or not args.index_code.isdigit():
        parser.error("品种须为字母，指数代码须为六位数字")
    args.index_code = args.index_code.zfill(6)
    if len(args.index_code) != 6 or args.end_day < args.start_day or not 0 <= args.index_precision <= 12:
        parser.error("日期、指数代码或指数精度无效")
    try:
        run_case(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"[耗时] main 总耗时={perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
