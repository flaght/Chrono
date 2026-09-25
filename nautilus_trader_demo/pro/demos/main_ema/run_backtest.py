"""CTP 品种动态主力 EMA 的离线 Bar 回测入口。只连接模拟交易端。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path

import pandas as pd
from bomber.backtest.config import BacktestEngineConfig
from bomber.config import LoggingConfig
from bomber.model import Venue
from bomber.model.identifiers import TraderId

from examples.black_sector.local_input import load_sector_research
from examples.main_ema.strategy import MainEmaConfig, MainEmaStrategy
from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import FixedInstrumentBarParser
from strategy import (
    ContractAssignment, CtpFuturesBasicProfile, DataBinding,
    DynamicExecutionRoute, MarketReferencePriceStore, MarketStreamBinding,
    NautilusMarketFeedAdapter, NautilusSimExecutionBackend,
    NetTargetOrderPlanner, PositionManager, PreTradeRiskManager, RiskLimits,
    RuntimeMode, ScheduledContractResolver, SimulationExecutionClient,
    UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)

DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "results"
DAY_NS = 86_400_000_000_000
MAX_COVERAGE_GAP = timedelta(days=14)
VENUE_ALIASES = {
    "XSGE": "SHFE", "SHFE": "SHFE",
    "XDCE": "DCE", "DCE": "DCE",
    "XZCE": "CZCE", "CZCE": "CZCE",
    "XSIE": "INE", "INE": "INE",
}


def _day_ns(value: object) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("合约挂牌/到期日不能为空")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.tz_convert("UTC").value)


def _bar_path(root: Path, symbol: str, day: date) -> Path:
    matches = tuple(path for path in root.rglob(f"*_{day:%Y%m%d}.feather")
                    if path.name.lower() == f"{symbol.lower()}_{day:%Y%m%d}.feather")
    if len(matches) != 1:
        raise FileNotFoundError(f"{day}/{symbol}需要唯一真实合约 Bar，找到 {len(matches)} 个")
    return matches[0]


def _venue_for_product(basic: pd.DataFrame, product: str) -> str:
    rows = basic.loc[basic["code"].astype(str).str.strip().str.upper() == product]
    if rows.empty:
        raise ValueError(f"fut_basic 没有品种 {product}")
    venues = {VENUE_ALIASES.get(str(value).strip().upper()) for value in rows["exchangeCD"]}
    if None in venues or len(venues) != 1:
        raise ValueError(f"{product} 的交易所不受支持或不唯一: {venues}")
    return venues.pop()


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
                product: str, symbol: str):
    rows = basic.loc[basic["symbol"].astype(str).str.strip().str.lower() == symbol]
    if len(rows) != 1:
        raise ValueError(f"fut_basic 需要唯一合约记录: {symbol}，实际 {len(rows)}")
    row = rows.iloc[0]
    if str(row["code"]).strip().upper() != product:
        raise ValueError(f"{symbol} 与品种 {product} 不匹配")
    if VENUE_ALIASES.get(str(row["exchangeCD"]).strip().upper()) != str(profile.venue):
        raise ValueError(f"{symbol} 与交易所 {profile.venue} 不匹配")
    multiplier = Decimal(str(row["contMultNum"]))
    if pd.isna(row["minChgPriceNum"]):
        raise ValueError(f"{symbol} 的 minChgPriceNum 为空")
    increment = Decimal(str(row["minChgPriceNum"]))
    if not multiplier.is_finite() or multiplier <= 0 or not increment.is_finite() or increment <= 0:
        raise ValueError(f"{symbol} 的乘数或最小变动价位无效")
    precision = max(0, -increment.normalize().as_tuple().exponent)
    instrument = profile.make_instrument(
        symbol, underlying=product.lower(), price_precision=precision,
        price_increment=increment, multiplier=multiplier,
        activation_ns=_day_ns(row["listDate"]),
        expiration_ns=_day_ns(row["lastTradeDate"]) + DAY_NS,
        margin_init=Decimal("0.10"), margin_maint=Decimal("0.08"),
    )
    meta = InstrumentMeta(
        instrument.id, price_precision=precision, size_precision=0,
        price_increment=increment, multiplier=multiplier,
        currency="CNY", exchange=str(profile.venue),
    )
    return instrument, meta, multiplier


def _write_reports(*, report_dir: Path, backend, backend_result,
                   strategy: MainEmaStrategy, days: tuple[date, ...],
                   product: str, venue: Venue,
                   orders: pd.DataFrame, fills: pd.DataFrame,
                   positions: pd.DataFrame, tearsheet: bool) -> None:
    """把可核对的原始报表与绩效指标保存到本次回测目录。"""
    report_dir = report_dir / str(backend_result.run_id)
    report_dir.mkdir(parents=True, exist_ok=True)
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


def run_case(*, product: str, start_day: date, end_day: date, bars_dir: Path,
             contract_struct: Path, fut_basic: Path,
             fast: int, slow: int,
             quantity: Decimal, require_fills: bool = False,
             log_level: str = "WARNING",
             report_dir: Path = DEFAULT_REPORT_DIR,
             tearsheet: bool = True,
             max_notional: Decimal = Decimal("1000000")) -> None:
    product = product.strip().upper()
    if not product.isalpha():
        raise ValueError("--product 须为品种字母代码，例如 RB、I、HC")
    if end_day < start_day:
        raise ValueError("日期范围无效")
    if not max_notional.is_finite() or max_notional <= 0:
        raise ValueError("最大名义金额须为正且有限")
    loaded = load_sector_research(
        bars_dir=bars_dir, contract_struct_path=contract_struct,
        signal_products=(product,), signal_role="main",
        execution_product=product, execution_role="main",
    )
    ends = dict(loaded.day_end_ns)
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
    assignments = tuple(loaded.store.snapshot(ends[day]) for day in days)
    symbols = {row.instrument(product, "main") for row in assignments}
    for previous, current in zip(assignments, assignments[1:]):
        if previous.instrument(product, "main") != current.instrument(product, "main"):
            symbols.add(previous.instrument(product, "main"))
    basic = pd.read_feather(fut_basic)
    required = {"code", "symbol", "exchangeCD", "contMultNum", "minChgPriceNum",
                "listDate", "lastTradeDate"}
    if required - set(basic.columns):
        raise ValueError(f"fut_basic 缺少列: {sorted(required - set(basic.columns))}")
    venue = Venue(_venue_for_product(basic, product))
    config = MainEmaConfig(
        product=product, venue=str(venue), fast_period=fast,
        slow_period=slow, quantity=quantity,
    )

    backend = NautilusSimExecutionBackend(
        f"{product.lower()}-main-ema-sim",
        BacktestEngineConfig(
            trader_id=TraderId(f"{product}-MAIN-EMA-001"),
            logging=LoggingConfig(log_level=log_level),
            run_analysis=True,
        ),
    )
    profile = CtpFuturesBasicProfile(
        starting_balance=Decimal("1000000"),
        commission_per_contract=Decimal(1), venue=venue,
    )
    backend.add_profile(profile)
    feed = FileReplayFeed(f"{product}_MAIN_EMA_REPLAY")
    instruments = {}
    multipliers = {}
    for symbol in sorted(symbols):
        instrument, meta, multiplier = _instrument(basic, profile, product, symbol)
        instruments[symbol] = instrument.id
        multipliers[instrument.id] = multiplier
        backend.add_instrument(instrument)
        feed.register_instrument(meta)

    for index, (day, row) in enumerate(zip(days, assignments)):
        needed = {row.instrument(product, "main")}
        if index and assignments[index - 1].instrument(product, "main") != row.instrument(product, "main"):
            needed.add(assignments[index - 1].instrument(product, "main"))
        for symbol in sorted(needed):
            path = _bar_path(bars_dir, symbol, day)
            feed.add_bar_feather(
                path, FixedInstrumentBarParser(
                    instruments[symbol], timestamp=_timestamp_column(path),
                ),
            )

    resolver = ScheduledContractResolver(tuple(
        ContractAssignment(config.target_key, instruments[row.instrument(product, "main")],
                           row.effective_ns, row.available_ns, revision)
        for revision, row in enumerate(assignments, 1)
    ))
    positions = PositionManager()
    prices = MarketReferencePriceStore()
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
        f"{product.lower()}-main-ema-clock", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE")
              for item in instruments.values()), manage_lifecycle=False,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
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
    try:
        result = runtime.run()
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
        _write_reports(
            report_dir=report_dir, backend=backend,
            backend_result=result.backend_result, strategy=strategy, days=days,
            product=product, venue=venue,
            orders=orders, fills=fills, positions=positions_report,
            tearsheet=tearsheet,
        )
    finally:
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="CTP 品种动态主力 EMA 离线回测")
    parser.add_argument("--product", required=True, help="品种代码，例如 RB、I、HC")
    parser.add_argument("--start-day", type=date.fromisoformat, required=True)
    parser.add_argument("--end-day", type=date.fromisoformat, required=True)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 KLINE_DIR")
    parser.add_argument("--contract-struct", type=Path, help="覆盖 ROLE_DIR 中的合约角色表")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 ROLE_DIR 中的合约基础表")
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
    missing = [name for name in ("ROLE_DIR", "KLINE_DIR") if not os.environ.get(name, "").strip()]
    if missing:
        parser.error(f"请先设置环境变量: {', '.join(missing)}")
    role_root = Path(os.environ["ROLE_DIR"]).expanduser()
    bars_root = Path(os.environ["KLINE_DIR"]).expanduser()
    run_case(product=args.product, start_day=args.start_day, end_day=args.end_day,
             bars_dir=args.bars_dir or bars_root,
             contract_struct=args.contract_struct or role_root / "fut_contract_data.feather",
             fut_basic=args.fut_basic or role_root / "fut_basic.feather",
             fast=args.fast, slow=args.slow, quantity=args.quantity,
             require_fills=args.require_fills, log_level=args.log_level,
             report_dir=args.report_dir, tearsheet=not args.no_tearsheet,
             max_notional=args.max_notional)


if __name__ == "__main__":
    main()
