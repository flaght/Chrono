"""Delta 双腿买入回测：公共数据准备、模拟撮合与选约/交易报表。"""
import argparse
from dataclasses import asdict
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from delta_signal import Config, Option
else:
    from .delta_signal import Config, Option

from dotenv import load_dotenv
from bomber.framework.dataprep.bars import PreparedFrameReader
from bomber.framework.dataprep.catalog import inventory
from bomber.framework.dataprep.contracts import BarFileKey, BarReadSpec
from bomber.framework.dataprep.metadata import load_options_basic, load_cffex_futures, positive
from bomber.framework.dataprep.paths import resolve_option_args
from bomber.framework.dataprep.scenarios.option_chain import plan_option_chain, prepare_option_chain
from bomber.framework.dataprep.session import input_session, write_input_reports

load_dotenv()
PAIRS = {"IO": ("IF", "000300"), "MO": ("IM", "000852"), "HO": ("IH", "000016")}


def _start_phase(name):
    print(f"[耗时] 开始：{name}", flush=True)
    return perf_counter()


def _finish_phase(name, started):
    print(f"[耗时] {name}完成，耗时={perf_counter() - started:.3f}s", flush=True)


def load_options(path, product, index_code):
    """校验条款与 tickNum，保留完整静态全集供期限选择。"""
    specs = load_options_basic(path, product, index_code)
    return {key: Option(key, spec.option_kind, float(spec.strike), spec.listed,
        spec.last_trade, float(spec.tick), float(spec.multiplier), spec.month)
        for key, spec in specs.items()}


def prepare_inputs(args, config):
    """准备可能交易的合约及其后续行情，不能按当天开仓期限丢掉旧持仓腿。"""
    phase = _start_phase("期权与期货基础条款加载")
    options = load_options(args.opt_basic, args.product, args.index_code)
    futures = load_cffex_futures(args.fut_basic, args.future_product, require_execution=False) if args.model == "black76" else {}
    _finish_phase("期权与期货基础条款加载", phase)
    phase = _start_phase("行情目录扫描与交易输入计划")
    paths = inventory(args.opt_dir, args.start_day, args.end_day, lambda key: key in options)
    indexes = inventory(args.index_dir, args.start_day, args.end_day, lambda key: key == args.index_code)
    future_paths = inventory(args.fut_dir, args.start_day, args.end_day, lambda key: key in futures) if futures else {}
    days = tuple(sorted({day for day, key in paths} | {day for day, key in indexes}))
    if not days:
        raise ValueError("请求区间没有期权和指数行情")
    needed, used_futures, tradable = {}, {}, set()
    for day in days:
        if (day, args.index_code) not in indexes:
            raise FileNotFoundError(f"{day}: 缺少指数{args.index_code}行情")
        if not any(file_day == day for file_day, key in paths):
            raise FileNotFoundError(f"{day}: 缺少{args.product}期权链")
        needed[day, args.index_code] = "index"
        for file_day, key in paths:
            info = options[key]
            dte = (info.expiry - day).days
            if file_day != day or info.kind not in config.sides or not info.listed <= day <= info.expiry:
                continue
            if dte < config.min_remaining_days:
                continue
            if config.expiry_month:
                if info.month != config.expiry_month:
                    continue
            elif not config.dte_min <= dte <= config.dte_max:
                continue
            tradable.add(key)
            needed[day, key] = "option"
            if args.model == "black76":
                future = args.future_product + info.month[2:]
                row = futures.get(future)
                if row is not None and row["listDate"] <= day <= row["lastTradeDate"]:
                    if (day, future) not in future_paths:
                        raise FileNotFoundError(f"{day}: 已上市同月期货{future}缺行情")
                    needed[day, future] = "future"
                    used_futures[info.month] = future
    # 曾满足开仓条件的期权后续即便退出期限窗口，也须加载其实际可用行情。
    for (day, key) in paths:
        if key in tradable and options[key].listed <= day <= options[key].expiry:
            needed[day, key] = "option"
    plan = plan_option_chain((BarFileKey(kind, key, day)
        for (day, key), kind in needed.items()), requested=(args.start_day, args.end_day))
    roots = {"option": args.opt_dir, "index": args.index_dir}
    if args.model == "black76":
        roots["future"] = args.fut_dir
    # 交易期权必须通过 OHLCV 和持仓量严格校验，禁止用研究替代价格撮合。
    specs = {"option": BarReadSpec(required_fields=("open", "high", "low", "close", "volume", "open_interest"),
        timestamp_label=args.bar_timestamp, trading_day_policy="day_session"),
        "index": BarReadSpec(required_fields=("close",), timestamp_label=args.bar_timestamp,
            trading_day_policy="day_session", value_policy="close_strict"),
        "future": BarReadSpec(required_fields=("close",), timestamp_label=args.bar_timestamp,
            trading_day_policy="day_session", value_policy="close_strict")}
    _finish_phase("行情目录扫描与交易输入计划", phase)
    phase = _start_phase("期权交易与标的信号行情加载校验")
    bundle = prepare_option_chain(roots, plan, specs=specs)
    _finish_phase("期权交易与标的信号行情加载校验", phase)
    return options, futures, used_futures, days, bundle


def _contract_info(option):
    """将研究条款转成合约工厂需要的精确价位与乘数。"""
    return SimpleNamespace(symbol=option.symbol, strike=option.strike,
        tick=Decimal(str(option.tick)), multiplier=Decimal(str(option.multiplier)),
        list_day=option.listed, last_day=option.expiry)


@input_session
def run_case(args):
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
    from bomber.framework.trader.instrument_factory import instrument_meta, make_option
    if __package__ in (None, ""):
        from strategy import OptionDeltaStrategy, SnapshotParser
    else:
        from .strategy import OptionDeltaStrategy, SnapshotParser

    started = perf_counter()
    phase = _start_phase("输入路径解析与配置校验")
    args = resolve_option_args(args, require_futures=args.model == "black76")
    config = Config(product=args.product, index_code=args.index_code, future_product=args.future_product,
        model=args.model, sides=tuple(args.sides.split(",")), delta_min=args.delta_min, delta_max=args.delta_max,
        target_delta=args.target_delta, fallback_min=args.fallback_min, fallback_max=args.fallback_max,
        allow_fallback=not args.no_fallback, dte_min=args.dte_min, dte_max=args.dte_max, target_dte=args.target_dte,
        min_remaining_days=args.min_remaining_days, expiry_month=args.expiry_month,
        select_slots=tuple(args.select_slots.split(",")), rate=args.rate, dividend_yield=args.dividend_yield,
        max_quote_age_seconds=args.max_quote_age_seconds, min_open_interest=args.min_open_interest,
        min_volume=args.min_volume, quantity=args.quantity, close_remaining_days=args.close_remaining_days,
        flatten_slot=args.flatten_slot, max_market_age_seconds=args.max_market_age_seconds)
    if set(config.sides) != {"C", "P"}:
        raise ValueError("双腿买入回测必须同时启用 C 和 P")
    for name in ("starting_balance", "max_notional", "option_margin_init", "option_margin_maint"):
        positive(getattr(args, name), name)
    if args.option_margin_maint > args.option_margin_init or args.option_margin_init > 1:
        raise ValueError("期权保证金比例无效")
    if not args.commission.is_finite() or args.commission < 0:
        raise ValueError("手续费须为有限非负数")
    _finish_phase("输入路径解析与配置校验", phase)
    options, futures, used_futures, days, bundle = prepare_inputs(args, config)
    # 信号期限选择仍使用静态全集，执行账户仅注册确实准备过行情的期权。
    traded_keys = sorted({source.result.key.symbol for source in bundle.sources
                         if source.result.key.asset_kind == "option"})
    if not traded_keys:
        raise ValueError("没有符合开仓期限且具备行情的期权；检查期限参数及输入文件")
    final_index = next(source.result.frame for source in bundle.sources
        if source.result.key.asset_kind == "index" and source.result.key.trading_day == days[-1])
    cutoff = pd.Timestamp(f"{days[-1]} {config.flatten_slot}", tz="Asia/Shanghai")
    if final_index.datetime.max() < cutoff + pd.Timedelta(minutes=2):
        raise ValueError("末日指数在清仓时点后不足两分钟行情，请调早 --flatten-slot")
    print(f"交易回测: product={args.product} model={args.model} 日期={days[0]}..{days[-1]} "
          f"days={len(days)} contracts={len(traded_keys)} 每腿={config.quantity} mode=BUY_PAIR")
    phase = _start_phase("模拟账户与真实认购认沽合约注册")
    backend = NautilusSimExecutionBackend("option-delta-sim", BacktestEngineConfig(
        trader_id=TraderId("OPTION-DELTA-001"), logging=LoggingConfig(log_level=args.log_level), run_analysis=True))
    try:
        backend.add_profile(CtpFuturesBasicProfile(profile_id="option-delta-basic", venue=Venue("CFFEX"),
            starting_balance=args.starting_balance, commission_per_contract=args.commission))
        feed = FileReplayFeed("OPTION_DELTA_REPLAY")
        instruments, multipliers, all_ids = {}, {}, {}
        for key in traded_keys:
            info = options[key]
            contract = make_option(_contract_info(info), args.option_margin_init, args.option_margin_maint,
                                   args.index_code, kind=info.kind, profile_id="OPTION_DELTA_BASIC")
            backend.add_instrument(contract)
            feed.register_instrument(instrument_meta(contract))
            instruments[key] = contract.id
            multipliers[contract.id] = Decimal(str(info.multiplier))
            all_ids[key] = contract.id
        # 指数和 Black76 期货只作为定价信号，没有交易路由或期货对冲。
        for key in sorted({source.result.key.symbol for source in bundle.sources} - set(instruments)):
            item = InstrumentId.from_str(key + (".XSHG" if key == args.index_code else ".CFFEX"))
            # Black76 期货只作信号，行情载体精度不依赖期货执行条款。
            tick = Decimal("0.0001")
            feed.register_instrument(InstrumentMeta(item, price_precision=max(0, -tick.normalize().as_tuple().exponent),
                size_precision=0, price_increment=tick, exchange=str(item.venue)))
            all_ids[key] = item
        _finish_phase("模拟账户与真实认购认沽合约注册", phase)
        phase = _start_phase("已准备行情注册到回放源")
        for source in bundle.sources:
            result = source.result
            feed.add_source(result.path, PreparedFrameReader(result.frame),
                SnapshotParser(all_ids[result.key.symbol], execution=result.key.asset_kind == "option"), "bar")
        _finish_phase("已准备行情注册到回放源", phase)
        phase = _start_phase("策略、执行路由与风控装配")
        positions, prices = PositionManager(), MarketReferencePriceStore()
        client = SimulationExecutionClient(backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
            risk_manager=PreTradeRiskManager(backend.backend_id, positions, prices,
                instrument_limits={item: RiskLimits(max_order_quantity=config.quantity,
                    max_abs_position=config.quantity, max_order_notional=args.max_notional,
                    max_abs_position_notional=args.max_notional,
                    max_market_age_ns=config.max_market_age_seconds * 1_000_000_000,
                    contract_multiplier=multipliers[item]) for item in instruments.values()}))
        strategy = OptionDeltaStrategy("option-delta", config, options, used_futures,
                                       instruments=instruments, final_day=days[-1])
        runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
        runner.add_market_observer(prices)
        runner.add_data_feed("option-bars", feed)
        runner.add_execution_client(client)
        runner.add_strategy(strategy, data_bindings=tuple(DataBinding(key, "option-bars", item,
            DataType.CUSTOM_BAR, "1-MINUTE") for key, item in all_ids.items()),
            execution_routes=tuple(ExecutionRoute(str(item), backend.backend_id, item) for item in instruments.values()))
        adapter = NautilusMarketFeedAdapter("option-delta-clock", feed, backend,
            tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE") for item in instruments.values()), manage_lifecycle=False)
        runtime = UnifiedHistoricalRuntime("option-delta-runtime", runner, adapter)
        _finish_phase("策略、执行路由与风控装配", phase)
        try:
            phase = _start_phase("运行回测")
            result = runtime.run()
            _finish_phase("运行回测", phase)
            phase = _start_phase("交易与选约报表写出及结果校验")
            reports = backend.engine.trader
            orders, fills = reports.generate_orders_report(), reports.generate_fills_report()
            output = Path(args.report_dir) / str(result.backend_result.run_id)
            output.mkdir(parents=True, exist_ok=True)
            write_input_reports(output)
            orders.to_csv(output / "orders.csv")
            fills.to_csv(output / "fills.csv")
            reports.generate_positions_report().to_csv(output / "positions.csv")
            reports.generate_account_report(Venue("CFFEX")).to_csv(output / "account.csv")
            for filename, rows, columns in (
                ("selections.csv", strategy.selections, ["date", "slot", "symbol", "kind", "month", "delta", "close"]),
                ("decisions.csv", strategy.records, ["date", "slot", "status", "count"]),
                ("candidates.csv", strategy.audits, ["date", "slot", "symbol", "reason", "selected"]),
                ("signals.csv", strategy.signals, ["signal_ns", "request_ns", "action"])):
                frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
                frame.to_csv(output / filename, index=False)
            remaining = {key: {"position": str(strategy.account_position(str(item))),
                              "working": str(strategy.working_quantity(str(item)))}
                for key, item in instruments.items()
                if strategy.account_position(str(item)) or strategy.working_quantity(str(item))}
            errors = [str(value) for value in client.report_errors]
            expected = {(day, slot) for day in days for slot in config.select_slots}
            actual = {(date.fromisoformat(row["date"]), row["slot"]) for row in strategy.records}
            missing = sorted(expected - actual)
            summary = {"mode": "BUY_PAIR", "research_only": False, "orders": len(orders), "fills": len(fills),
                "requested": [str(args.start_day), str(args.end_day)], "actual": [str(days[0]), str(days[-1])],
                "config": asdict(config), "bars": result.replay_summary.bars,
                "snapshots": result.replay_summary.custom_bars, "decisions": len(strategy.records),
                "selections": len(strategy.selections), "submissions": strategy.submissions,
                "missing_slots": [(str(day), slot) for day, slot in missing],
                "flatten_requested": strategy.flatten_requested, "remaining_positions_or_working": remaining,
                "report_errors": errors, "price_source": "close", "hedge": "NONE",
                "simulation": "NETTING/BAR/FIXED_FEE/PREMIUM_PERCENT_MARGIN",
                "order_book_checks": False, "multi_leg_atomic": False}
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
            print(f"decisions={len(strategy.records)} selections={len(strategy.selections)} "
                  f"orders={len(orders)} fills={len(fills)} 报表={output}")
            if remaining or errors or not strategy.flatten_requested:
                raise RuntimeError(f"回测未完成清仓：残留={remaining} 回报异常={errors} 清仓触发={strategy.flatten_requested}；报告已保存")
            if missing:
                raise ValueError(f"部分选约时点没有行情: {missing[:10]}；报告已保存")
            if args.require_selection and not any(row["status"] == "SELECTED" for row in strategy.records):
                raise ValueError("未选出完整 Call/Put 双腿；报告已保存")
            if args.require_fills and fills.empty:
                raise RuntimeError("要求成交，但没有成交；检查 candidates.csv 与 signals.csv")
            _finish_phase("交易与选约报表写出及结果校验", phase)
            if not args.no_tearsheet:
                phase = _start_phase("绩效图生成")
                try:
                    from bomber.analysis.tearsheet import create_tearsheet
                    create_tearsheet(engine=backend.engine, output_path=str(output / "tearsheet.html"), title="Option Delta Buy Pair Backtest")
                except ImportError as exc:
                    print(f"绩效图依赖不可用: {exc}；CSV/JSON已保存")
                _finish_phase("绩效图生成", phase)
        finally:
            phase = _start_phase("运行器清理")
            runtime.stop()
            _finish_phase("运行器清理", phase)
    finally:
        if not backend.is_disposed:
            backend.stop()
        print(f"[耗时] run_case 总耗时={perf_counter() - started:.3f}s", flush=True)


@input_session
def main():
    p=argparse.ArgumentParser(description="中金所Delta选约、Call/Put双腿买入及完整交易回测")
    p.add_argument("--product",choices=tuple(PAIRS),default="MO")
    p.add_argument("--index-code")
    p.add_argument("--future-product")
    p.add_argument("--start-day",type=date.fromisoformat,required=True)
    p.add_argument("--end-day",type=date.fromisoformat,required=True)
    p.add_argument("--bar-timestamp",choices=("start","end"),default="end")
    for name in ("data-root","opt-dir","index-dir","fut-dir","opt-basic","fut-basic"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--model",choices=("bs","black76"),default="bs")
    p.add_argument("--sides",default="C,P")
    p.add_argument("--expiry-month")
    p.add_argument("--select-slots",default="13:58")
    for name,default in (("delta-min",.25),("delta-max",.30),("target-delta",.275),("fallback-min",.225),("fallback-max",.325),
        ("target-dte",32.5),("rate",.02),("dividend-yield",0.),
        ("min-open-interest",1.),("min-volume",0.)):
        p.add_argument("--"+name,type=float,default=default)
    for name,default in (("dte-min",20),("dte-max",45),("min-remaining-days",7)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--max-quote-age-seconds",type=int,default=60,
        help="兼容原参数名：仅用于完整分钟帧最大迟到秒数，不检验成交价更新时间")
    p.add_argument("--no-fallback",action="store_true")
    p.add_argument("--require-selection",action="store_true")
    p.add_argument("--quantity",type=int,default=1,help="每条期权腿的目标手数，默认1")
    p.add_argument("--close-remaining-days",type=int,default=2,help="剩余自然日不超过此值时平整个组合")
    p.add_argument("--flatten-slot",default="14:50",help="回测末日清仓时点")
    p.add_argument("--max-market-age-seconds",type=int,default=300)
    p.add_argument("--starting-balance",type=Decimal,default=Decimal("1000000"))
    p.add_argument("--commission",type=Decimal,default=Decimal("1"),help="每手固定手续费")
    p.add_argument("--max-notional",type=Decimal,default=Decimal("10000000"))
    p.add_argument("--option-margin-init",type=Decimal,default=Decimal("1"))
    p.add_argument("--option-margin-maint",type=Decimal,default=Decimal("1"))
    p.add_argument("--log-level",choices=("ERROR","WARNING","INFO","DEBUG"),default="WARNING")
    p.add_argument("--require-fills",action="store_true")
    p.add_argument("--no-tearsheet",action="store_true")
    p.add_argument("--report-dir",type=Path,default=Path(__file__).resolve().parent/"results")
    args=p.parse_args()
    pair=PAIRS[args.product]
    args.future_product=(args.future_product or pair[0]).strip().upper()
    args.index_code=(args.index_code or pair[1]).strip().zfill(6)
    args.sides=",".join(item.strip().upper() for item in args.sides.split(","))
    args.select_slots=",".join(item.strip() for item in args.select_slots.split(","))
    if args.end_day<args.start_day or not args.index_code.isdigit() or len(args.index_code)!=6: p.error("日期或指数代码无效")
    try: run_case(args)
    except (ValueError,FileNotFoundError) as exc: p.error(str(exc))


if __name__=="__main__": main()
