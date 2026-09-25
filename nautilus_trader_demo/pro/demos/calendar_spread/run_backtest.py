"""可指定品种的两期限真实合约价差模拟回测。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from time import perf_counter

import pandas as pd
from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from dotenv import load_dotenv 
load_dotenv()

from demos.calendar_spread.spread_strategy import (
    CalendarSelection, CalendarSpreadConfig, CalendarSpreadStrategy,
)
from demos.calendar_spread.contract_input import (
    bar_paths, instrument, timestamp_column, venue,
)
from demos.calendar_spread.selection import ROLE_COLUMNS, ROLE_ORDER, available_pair
from market.basic.base import DataType
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import FixedInstrumentBarParser
from strategy import (
    CtpFuturesBasicProfile, DataBinding, ExecutionRoute, MarketReferencePriceStore,
    MarketStreamBinding, NautilusMarketFeedAdapter, NautilusSimExecutionBackend,
    NetTargetOrderPlanner, PositionManager, PreTradeRiskManager, RiskLimits,
    RuntimeMode, SimulationExecutionClient, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)

FILE_PATTERN = re.compile(r"^([a-z]+)\d+_(\d{8})\.feather$", re.IGNORECASE)
MAX_COVERAGE_GAP = timedelta(days=14)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"


def _event_ns(value: object) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Bar 时间不能为空")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Shanghai")
    return int(stamp.tz_convert("UTC").value)


def _first_ns(path: Path) -> int:
    column = timestamp_column(path)
    frame = pd.read_feather(path, columns=[column])
    if frame.empty:
        raise ValueError(f"Bar 文件为空: {path}")
    return min(_event_ns(value) for value in frame[column])


def _bar_inventory(bars_dir: Path, product: str,
                   start_day: date, end_day: date) -> dict[date, set[str]]:
    available: dict[date, set[str]] = {}
    for path in bars_dir.rglob("*.feather"):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None or match.group(1).upper() != product:
            continue
        label = match.group(2)
        day = date.fromisoformat(f"{label[:4]}-{label[4:6]}-{label[6:]}")
        if start_day <= day <= end_day:
            symbol = path.name.split("_", 1)[0].lower()
            available.setdefault(day, set()).add(symbol)
    return available


def _select_roles(roles: pd.DataFrame, days: tuple[date, ...], product: str,
                  config: CalendarSpreadConfig, available: dict[date, set[str]],
                  missing_role_policy: str) -> tuple[tuple[tuple[date, str, str], ...], tuple[str, ...]]:
    required = {"trade_date", "code", ROLE_COLUMNS[config.near_role],
                ROLE_COLUMNS[config.far_role]}
    if required - set(roles.columns):
        raise ValueError(f"角色表缺少列: {sorted(required - set(roles.columns))}")
    rows = roles.loc[roles["code"].astype(str).str.strip().str.upper() == product].copy()
    if rows.empty:
        raise ValueError(f"角色表没有 {product}")
    rows["source_day"] = rows["trade_date"].map(lambda value: pd.Timestamp(value).date())
    rows = rows.sort_values("source_day").drop_duplicates("source_day", keep="last")
    chosen = []
    fallbacks = []
    near_index = ROLE_ORDER.index(config.near_role)
    far_index = ROLE_ORDER.index(config.far_role)
    if near_index >= far_index:
        raise ValueError("near-role 必须位于 far-role 之前：main、secondary、far")
    for day in days:
        prior = rows.loc[rows["source_day"] < day]
        if prior.empty:
            raise ValueError(f"{day} 没有上一交易日 {product} 角色记录")
        row = prior.iloc[-1]
        symbols = {role: str(row[column]).strip().lower()
                   for role, column in ROLE_COLUMNS.items() if column in rows.columns}
        near = symbols[config.near_role]
        far = symbols[config.far_role]
        pattern = rf"{re.escape(product)}\d+"
        if re.fullmatch(pattern, near, re.IGNORECASE) is None or re.fullmatch(pattern, far, re.IGNORECASE) is None:
            raise ValueError(f"{day} 的 {product} 两期限角色无效: {near}, {far}")
        if near == far:
            raise ValueError(f"{day} 的两个期限角色指向同一合约: {near}")
        present = available[day]
        if near not in present or far not in present:
            if missing_role_policy == "raise":
                missing = [symbol for symbol in (near, far) if symbol not in present]
                raise FileNotFoundError(f"{day} 角色表合约缺少真实 Bar: {missing}")
            original = (near, far)
            replacement = available_pair(symbols, config.near_role, config.far_role, present)
            if replacement is None:
                raise FileNotFoundError(
                    f"{day} {product} 角色表期限 {original} 缺 Bar，且无法找到两个有 Bar 的替代期限"
                )
            near, far = replacement
            fallbacks.append(f"{day}: {original[0]}/{original[1]} -> {near}/{far}")
        chosen.append((day, near, far))
    return tuple(chosen), tuple(fallbacks)


def run_case(*, product: str, start_day: date, end_day: date, bars_dir: Path,
             contract_struct: Path, fut_basic: Path,
             near_role: str = "secondary", far_role: str = "far",
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
             tearsheet: bool = True, require_fills: bool = False) -> None:
    started = perf_counter()
    product = product.strip().upper()
    if not product.isalpha() or end_day < start_day:
        raise ValueError("品种代码或日期范围无效")
    if missing_role_policy not in {"raise", "next-available"}:
        raise ValueError("missing_role_policy 必须为 raise 或 next-available")
    config = CalendarSpreadConfig(
        near_role=near_role, far_role=far_role,
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
    available = _bar_inventory(bars_dir, product, start_day, end_day)
    days = tuple(sorted(available))
    if not days:
        raise ValueError(f"{start_day}..{end_day} 没有 {product} Bar")
    gaps = tuple((left, right) for left, right in zip(days, days[1:])
                 if right - left > MAX_COVERAGE_GAP)
    print(f"行情覆盖: product={product} 请求={start_day}..{end_day} 实际={days[0]}..{days[-1]} "
          f"交易日数={len(days)}", flush=True)
    if days[0] - start_day > MAX_COVERAGE_GAP or end_day - days[-1] > MAX_COVERAGE_GAP or gaps:
        raise ValueError(f"行情未覆盖请求区间；超过14天的缺口={gaps}")
    chosen, role_fallbacks = _select_roles(
        pd.read_feather(contract_struct), days, product, config,
        available, missing_role_policy,
    )
    if role_fallbacks:
        print(f"角色表合约缺 Bar，已使用当日有 Bar 的相邻角色（{len(role_fallbacks)} 天）："
              f"{role_fallbacks[:10]}", flush=True)
    required_bars = set()
    optional_old_bars = set()
    previous_pairs = []
    for index, (day, near, far) in enumerate(chosen):
        required_bars.update({(day, near), (day, far)})
        old_pair = None
        if index and (chosen[index - 1][1], chosen[index - 1][2]) != (near, far):
            old_pair = chosen[index - 1][1:]
            optional_old_bars.update((day, symbol) for symbol in old_pair)
        previous_pairs.append(old_pair)
    paths = bar_paths(bars_dir, required_bars, optional_old_bars)
    needed_by_day = tuple((day, {symbol for candidate_day, symbol in paths if candidate_day == day})
                          for day, _, _ in chosen)
    selections = tuple(
        CalendarSelection(
            day, min(_first_ns(paths[(day, near)]), _first_ns(paths[(day, far)])),
            near, far,
            old_pair_bars_available=(old_pair is None or all(
                (day, symbol) in paths for symbol in old_pair
            )),
        )
        for (day, near, far), old_pair in zip(chosen, previous_pairs)
    )
    missing_old = tuple((day, symbol) for (day, _, _), old_pair in zip(chosen, previous_pairs)
                        if old_pair for symbol in old_pair if (day, symbol) not in paths)
    if missing_old:
        print(f"换月旧合约缺 Bar（空仓时可跳过）: {missing_old[:10]}", flush=True)
    basic = pd.read_feather(fut_basic)
    required = {"symbol", "code", "exchangeCD", "contMultNum", "minChgPriceNum",
                "listDate", "lastTradeDate"}
    if required - set(basic.columns):
        raise ValueError(f"fut_basic 缺少列: {sorted(required - set(basic.columns))}")
    venue_name = venue(basic, product)
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
    for day, needed in needed_by_day:
        for symbol in sorted(needed):
            path = paths[(day, symbol)]
            feed.add_bar_feather(path, FixedInstrumentBarParser(
                instruments[symbol], timestamp=timestamp_column(path),
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
            ) for item in instruments.values()},
        ),
    )
    strategy = CalendarSpreadStrategy(
        f"{product.lower()}-calendar-spread", config, selections, instruments,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
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
    replay_started = perf_counter()
    print(f"回放装配耗时={replay_started - started:.1f}s，开始回放", flush=True)
    try:
        result = runtime.run()
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
        output = report_dir / str(result.backend_result.run_id)
        output.mkdir(parents=True, exist_ok=True)
        orders.to_csv(output / "orders.csv")
        fills.to_csv(output / "fills.csv")
        trader.generate_positions_report().to_csv(output / "positions.csv")
        trader.generate_account_report(Venue(venue_name)).to_csv(output / "account.csv")
        summary = (asdict(result.backend_result) if is_dataclass(result.backend_result)
                   else {"backend_result": str(result.backend_result)})
        summary["strategy"] = {
            "product": product, "venue": venue_name,
            "missing_role_policy": missing_role_policy,
            "role_fallbacks": role_fallbacks,
            "near_role": config.near_role, "far_role": config.far_role,
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
    finally:
        print(f"历史回放耗时={perf_counter() - replay_started:.1f}s", flush=True)
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="单品种两期限价差均值回归回测")
    parser.add_argument("--product", required=True, help="品种，例如 RB、HC、I、JM")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day", required=True, type=date.fromisoformat)
    parser.add_argument("--near-role", choices=("main", "secondary"), default="secondary")
    parser.add_argument("--far-role", choices=("secondary", "far"), default="far")
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 KLINE_DIR")
    parser.add_argument("--contract-struct", type=Path, help="覆盖 ROLE_DIR 中的角色表")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 ROLE_DIR 中的合约基础表")
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
    role_root = Path(os.environ["ROLE_DIR"]).expanduser() if os.environ.get("ROLE_DIR") else None
    bars_dir = args.bars_dir or (Path(os.environ["KLINE_DIR"]).expanduser()
                                 if os.environ.get("KLINE_DIR") else None)
    contract_struct = args.contract_struct or (role_root / "fut_contract_data.feather" if role_root else None)
    fut_basic = args.fut_basic or (role_root / "fut_basic.feather" if role_root else None)
    if bars_dir is None or contract_struct is None or fut_basic is None:
        parser.error("请设置 ROLE_DIR、KLINE_DIR，或分别传入 --bars-dir、--contract-struct、--fut-basic")
    run_case(product=args.product, start_day=args.start_day, end_day=args.end_day,
             bars_dir=bars_dir, contract_struct=contract_struct,
             fut_basic=fut_basic, near_role=args.near_role, far_role=args.far_role,
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


if __name__ == "__main__":
    main()
