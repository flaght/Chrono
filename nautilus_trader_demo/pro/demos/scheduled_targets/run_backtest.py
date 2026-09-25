"""外部目标 CSV + 独立计划时钟 + CTP 真实合约 Bar 的模拟回测。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
import json
import os
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from dotenv import load_dotenv 
load_dotenv()

import pandas as pd
from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from demos.scheduled_targets.contract_input import (
    bar_files, contract_rows, make_instrument, symbol_of, timestamp_column, venue_of,
)
from demos.scheduled_targets.local_input import load_target_csv
from demos.scheduled_targets.targets_strategy import ScheduledTargetStrategy
from market.basic.base import DataType
from market.replay.parsers.bar import FixedInstrumentBarParser
from strategy import (
    CtpFuturesBasicProfile, ExecutionRoute, MarketReferencePriceStore,
    MarketStreamBinding, NautilusMarketFeedAdapter, NautilusSimExecutionBackend,
    NetTargetOrderPlanner, PositionManager, PreTradeRiskManager, RiskLimits,
    RuntimeMode, SimulationExecutionClient, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)
from strategy.scheduling import TimedFileReplayFeed

DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"


def run_case(*, targets: Path, start_day: date, end_day: date, bars_dir: Path,
             fut_basic: Path, timezone_name: str = "Asia/Shanghai",
             starting_balance: Decimal = Decimal("1000000"),
             commission_per_contract: Decimal = Decimal(1),
             margin_init: Decimal = Decimal("0.10"),
             margin_maint: Decimal = Decimal("0.08"),
             max_quantity: Decimal = Decimal(100),
             max_notional: Decimal = Decimal("1000000"),
             max_market_age_seconds: int = 120,
             log_level: str = "WARNING", report_dir: Path = DEFAULT_REPORT_DIR,
             tearsheet: bool = True, require_fills: bool = False,
             allow_missed_slots: bool = False) -> None:
    started = perf_counter()
    if end_day < start_day:
        raise ValueError("日期范围无效")
    if any(not value.is_finite() or value <= 0 for value in
           (starting_balance, margin_init, margin_maint, max_quantity, max_notional)):
        raise ValueError("账户资金、保证金、数量和风控上限必须为有限正数")
    if (not commission_per_contract.is_finite() or commission_per_contract < 0
            or margin_init > 1 or margin_maint > margin_init
            or max_quantity != max_quantity.to_integral_value()
            or max_market_age_seconds < 1):
        raise ValueError("手续费、保证金比例、最大手数或行情时效无效")
    zone = ZoneInfo(timezone_name)
    schedule = load_target_csv(targets, timezone_name=timezone_name)
    outside = tuple(datetime.fromtimestamp(slot / 1_000_000_000, zone).date()
                    for slot in schedule.slots
                    if not start_day <= datetime.fromtimestamp(slot / 1_000_000_000, zone).date() <= end_day)
    if outside:
        raise ValueError(f"目标时点超出请求日期范围: {outside[:10]}")
    keys = {key for plan in schedule.plans for key in plan.targets}
    if any(abs(quantity) > max_quantity for plan in schedule.plans
           for quantity in plan.targets.values()):
        raise ValueError("计划目标手数超过 --max-quantity")
    rows = contract_rows(pd.read_feather(fut_basic), keys)
    identity = [(symbol_of(row).lower(), venue_of(row)) for row in rows.values()]
    if len(set(identity)) != len(identity):
        raise ValueError("目标文件中存在多个键指向同一真实合约；请统一 target_key 写法")
    files = bar_files(bars_dir, {symbol_of(row) for row in rows.values()}, start_day, end_day)
    venues = {venue_of(row) for row in rows.values()}
    backend = NautilusSimExecutionBackend(
        "scheduled-targets-sim", BacktestEngineConfig(
            trader_id=TraderId("SCHEDULED-TARGETS-001"),
            logging=LoggingConfig(log_level=log_level), run_analysis=True,
        ),
    )
    profiles = {}
    for venue_name in sorted(venues):
        profile = CtpFuturesBasicProfile(
            profile_id=f"scheduled-{venue_name.lower()}",
            starting_balance=starting_balance,
            commission_per_contract=commission_per_contract,
            venue=Venue(venue_name),
        )
        backend.add_profile(profile)
        profiles[venue_name] = profile
    feed = TimedFileReplayFeed(schedule.slots, "SCHEDULED_TARGETS_REPLAY")
    instruments = {}
    multipliers = {}
    for key, row in sorted(rows.items()):
        contract, meta, multiplier = make_instrument(
            row, profiles[venue_of(row)], margin_init, margin_maint,
        )
        backend.add_instrument(contract)
        feed.register_instrument(meta)
        instruments[key] = contract.id
        multipliers[contract.id] = multiplier
        for path in files[symbol_of(row).lower()]:
            feed.add_bar_feather(path, FixedInstrumentBarParser(
                contract.id, timestamp=timestamp_column(path),
            ))
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    client = SimulationExecutionClient(
        backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id, positions, prices,
            instrument_limits={item: RiskLimits(
                max_order_quantity=max_quantity * 2,
                max_abs_position=max_quantity,
                max_order_notional=max_notional,
                max_abs_position_notional=max_notional,
                max_market_age_ns=max_market_age_seconds * 1_000_000_000,
                contract_multiplier=multipliers[item],
            ) for item in instruments.values()},
        ),
    )
    strategy = ScheduledTargetStrategy("scheduled-targets", schedule)
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("scheduled-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy, data_bindings=(), time_feed_ids=("scheduled-bars",),
        execution_routes=tuple(ExecutionRoute(key, backend.backend_id, item)
                               for key, item in instruments.items()),
    )
    adapter = NautilusMarketFeedAdapter(
        "scheduled-targets-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE")
              for item in instruments.values()), manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime("scheduled-targets-runtime", runner, adapter)
    print(f"计划={len(schedule.slots)}时点 合约={len(instruments)} Bar文件="
          f"{sum(map(len, files.values()))} 装配耗时={perf_counter() - started:.1f}s",
          flush=True)
    replay_started = perf_counter()
    try:
        result = runtime.run()
        trader = backend.engine.trader
        orders = trader.generate_orders_report()
        fills = trader.generate_fills_report()
        strategy.finalize_audit()
        statuses = {status: sum(item.status == status for item in strategy.audit)
                    for status in ("SUBMITTED", "MISSED", "NOT_REACHED")}
        print(f"bars={result.replay_summary.bars} slots={len(schedule.slots)} "
              f"audit={statuses} orders={len(orders)} fills={len(fills)}")
        output = report_dir / str(result.backend_result.run_id)
        output.mkdir(parents=True, exist_ok=True)
        orders.to_csv(output / "orders.csv")
        fills.to_csv(output / "fills.csv")
        pd.DataFrame([asdict(item) for item in strategy.audit]).to_csv(
            output / "schedule_audit.csv", index=False,
        )
        trader.generate_positions_report().to_csv(output / "positions.csv")
        for venue_name in sorted(venues):
            trader.generate_account_report(Venue(venue_name)).to_csv(
                output / f"account_{venue_name}.csv"
            )
        summary = (asdict(result.backend_result) if is_dataclass(result.backend_result)
                   else {"backend_result": str(result.backend_result)})
        summary["strategy"] = {
            "targets_file": str(targets), "requested": [str(start_day), str(end_day)],
            "venues": sorted(venues), "instruments": {key: str(item) for key, item in instruments.items()},
            "bars": result.replay_summary.bars, "slots": len(schedule.slots),
            "audit": [asdict(item) for item in strategy.audit],
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
                                 title="Scheduled Targets Backtest")
            except ImportError as exc:
                print(f"绩效图未生成（{exc}）；CSV 和 JSON 已保存")
            else:
                print(f"交互绩效图: {chart.resolve()}")
        if client.report_errors:
            raise AssertionError(f"模拟执行回报异常: {client.report_errors}")
        if not allow_missed_slots and statuses["SUBMITTED"] != len(schedule.slots):
            raise AssertionError(f"部分目标时点未触发，审计见 {output / 'schedule_audit.csv'}")
        if require_fills and fills.empty:
            raise AssertionError("本次要求成交，但没有模拟成交")
    finally:
        print(f"历史回放耗时={perf_counter() - replay_started:.1f}s", flush=True)
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="外部目标计划驱动的 CTP 期货回测")
    parser.add_argument("--targets", required=True, type=Path, help="目标 CSV 文件")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day", required=True, type=date.fromisoformat)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 KLINE_DIR")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 ROLE_DIR/fut_basic.feather")
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--starting-balance", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--commission-per-contract", type=Decimal, default=Decimal(1))
    parser.add_argument("--margin-init", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--margin-maint", type=Decimal, default=Decimal("0.08"))
    parser.add_argument("--max-quantity", type=Decimal, default=Decimal(100))
    parser.add_argument("--max-notional", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--max-market-age-seconds", type=int, default=120)
    parser.add_argument("--log-level", choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        default="WARNING")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--no-tearsheet", action="store_true")
    parser.add_argument("--require-fills", action="store_true")
    parser.add_argument("--allow-missed-slots", action="store_true")
    args = parser.parse_args()
    role_root = Path(os.environ["ROLE_DIR"]).expanduser() if os.environ.get("ROLE_DIR") else None
    bars_dir = args.bars_dir or (Path(os.environ["KLINE_DIR"]).expanduser()
                                 if os.environ.get("KLINE_DIR") else None)
    fut_basic = args.fut_basic or (role_root / "fut_basic.feather" if role_root else None)
    if bars_dir is None or fut_basic is None:
        parser.error("请设置 KLINE_DIR、ROLE_DIR，或传入 --bars-dir、--fut-basic")
    run_case(targets=args.targets, start_day=args.start_day, end_day=args.end_day,
             bars_dir=bars_dir, fut_basic=fut_basic, timezone_name=args.timezone,
             starting_balance=args.starting_balance,
             commission_per_contract=args.commission_per_contract,
             margin_init=args.margin_init, margin_maint=args.margin_maint,
             max_quantity=args.max_quantity, max_notional=args.max_notional,
             max_market_age_seconds=args.max_market_age_seconds,
             log_level=args.log_level, report_dir=args.report_dir,
             tearsheet=not args.no_tearsheet, require_fills=args.require_fills,
             allow_missed_slots=args.allow_missed_slots)


if __name__ == "__main__":
    main()
