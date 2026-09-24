"""Configurable CTP cross-sectional momentum backtest using real contract bars."""

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
from examples.cross_section.cross_strategy import MainCrossSectionMomentumStrategy
from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import FixedInstrumentBarParser
from trader import (
    CtpFuturesBasicProfile, DataBinding, ExecutionRoute, MarketReferencePriceStore,
    MarketStreamBinding, NautilusMarketFeedAdapter, NautilusSimExecutionBackend,
    NetTargetOrderPlanner, PositionManager, PreTradeRiskManager, RiskLimits,
    RuntimeMode, SimulationExecutionClient, UnifiedHistoricalRuntime, UnifiedStrategyRunner,
)

VENUE_ALIASES = {"XSGE": "SHFE", "SHFE": "SHFE", "XDCE": "DCE", "DCE": "DCE",
                 "XZCE": "CZCE", "CZCE": "CZCE", "XSIE": "INE", "INE": "INE"}
DAY_NS = 86_400_000_000_000


def _ns(value: object) -> int:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("合约挂牌/到期日不能为空")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.tz_convert("UTC").value)


def _instrument(basic: pd.DataFrame, profile: CtpFuturesBasicProfile,
                product: str, symbol: str, margin_init: Decimal,
                margin_maint: Decimal):
    rows = basic.loc[basic["symbol"].astype(str).str.strip().str.lower() == symbol.lower()]
    if len(rows) != 1:
        raise ValueError(f"fut_basic 需要唯一合约记录: {symbol}")
    row = rows.iloc[0]
    if str(row["code"]).strip().upper() != product or VENUE_ALIASES.get(
        str(row["exchangeCD"]).strip().upper()
    ) != str(profile.venue):
        raise ValueError(f"{symbol} 的品种或交易所不匹配")
    increment = Decimal(str(row["minChgPriceNum"]))
    multiplier = Decimal(str(row["contMultNum"]))
    if not increment.is_finite() or increment <= 0 or not multiplier.is_finite() or multiplier <= 0:
        raise ValueError(f"{symbol} 的价格步长或乘数无效")
    precision = max(0, -increment.normalize().as_tuple().exponent)
    instrument = profile.make_instrument(
        symbol, underlying=product.lower(), price_precision=precision,
        price_increment=increment, multiplier=multiplier,
        activation_ns=_ns(row["listDate"]), expiration_ns=_ns(row["lastTradeDate"]) + DAY_NS,
        margin_init=margin_init, margin_maint=margin_maint,
    )
    meta = InstrumentMeta(
        instrument.id, price_precision=precision, size_precision=0,
        price_increment=increment, multiplier=multiplier,
        currency="CNY", exchange=str(profile.venue),
    )
    return instrument, meta, multiplier


def _timestamp_column(path: Path) -> str:
    import pyarrow as pa
    for name in ("datetime", "timestamp"):
        try:
            pd.read_feather(path, columns=[name])
        except (KeyError, ValueError, pa.ArrowInvalid):
            continue
        return name
    raise ValueError(f"{path} 缺少 datetime/timestamp 列")


def _selected_products(basic: pd.DataFrame, products: tuple[str, ...]) -> dict[str, str]:
    required = {"symbol", "code", "exchangeCD", "contMultNum", "minChgPriceNum",
                "listDate", "lastTradeDate"}
    if required - set(basic.columns):
        raise ValueError(f"fut_basic 缺少列: {sorted(required - set(basic.columns))}")
    if len(products) < 2 or len(set(products)) != len(products) or any(
        not item.isalpha() or item != item.upper() for item in products
    ):
        raise ValueError("--products 须指定至少两个不重复的品种字母代码")
    result = {}
    for product in products:
        rows = basic.loc[basic["code"].astype(str).str.strip().str.upper() == product]
        if rows.empty:
            raise ValueError(f"fut_basic 没有品种 {product}")
        venues = {VENUE_ALIASES.get(str(value).strip().upper()) for value in rows["exchangeCD"]}
        if None in venues or len(venues) != 1:
            raise ValueError(f"{product} 的交易所不受支持或不唯一: {venues}")
        result[product] = venues.pop()
    return result


def _files(root: Path, symbol: str, start: date, end: date) -> dict[date, Path]:
    result = {}
    for path in root.rglob("*.feather"):
        stem, separator, day_text = path.stem.rpartition("_")
        if not separator or stem.lower() != symbol.lower() or len(day_text) != 8 or not day_text.isdigit():
            continue
        day = date.fromisoformat(f"{day_text[:4]}-{day_text[4:6]}-{day_text[6:]}")
        if start <= day <= end:
            if day in result:
                raise ValueError(f"重复 Bar 文件: {symbol}/{day}")
            result[day] = path
    if not result:
        raise FileNotFoundError(f"{root} 中 {start}..{end} 无 {symbol} Bar")
    return result


def run_case(*, products: tuple[str, ...], start_day: date, end_day: date,
             bars_dir: Path, contract_struct: Path, fut_basic: Path, lookback: int = 20,
             rebalance_interval: int = 5, target_notional: Decimal = Decimal("100000"),
             starting_balance: Decimal = Decimal("1000000"),
             commission_per_contract: Decimal = Decimal("1"),
             margin_init: Decimal = Decimal("0.10"),
             margin_maint: Decimal = Decimal("0.08"),
             max_notional: Decimal = Decimal("1000000"),
             log_level: str = "WARNING", report_dir: Path | None = None,
             tearsheet: bool = True, require_fills: bool = False) -> None:
    if end_day < start_day:
        raise ValueError("日期范围无效")
    if any(not value.is_finite() or value <= 0 for value in
           (target_notional, starting_balance, max_notional, margin_init, margin_maint)) or commission_per_contract < 0:
        raise ValueError("资金、名义金额和风控上限须为正，手续费不能为负")
    if margin_init > 1 or margin_maint > margin_init:
        raise ValueError("保证金比例须满足 0 < margin_maint <= margin_init <= 1")
    products = tuple(item.strip().upper() for item in products)
    basic = pd.read_feather(fut_basic)
    venues = _selected_products(basic, products)
    loaded = load_sector_research(
        bars_dir=bars_dir, contract_struct_path=contract_struct,
        signal_products=products, signal_role="main",
        execution_product=products[0], execution_role="main",
    )
    ends = dict(loaded.day_end_ns)
    days = tuple(day for day in sorted(ends) if start_day <= day <= end_day)
    if not days:
        raise ValueError("请求区间没有可用的主力 Bar")
    if days[0] - start_day > timedelta(days=14) or end_day - days[-1] > timedelta(days=14) or any(
        right - left > timedelta(days=14) for left, right in zip(days, days[1:])
    ):
        raise ValueError(f"行情未覆盖请求区间 {start_day}..{end_day}；实际仅 {days[0]}..{days[-1]}")
    assignments = tuple(loaded.store.snapshot(ends[day]) for day in days)
    selected: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(assignments):
        for product in products:
            selected[row.instrument(product, "main").lower()] = (product, venues[product])
            if index and assignments[index - 1].instrument(product, "main") != row.instrument(product, "main"):
                selected[assignments[index - 1].instrument(product, "main").lower()] = (product, venues[product])
    files = {symbol: _files(bars_dir, symbol, start_day, end_day) for symbol in selected}
    for index, (day, row) in enumerate(zip(days, assignments)):
        for product in products:
            symbols = {row.instrument(product, "main").lower()}
            if index and assignments[index - 1].instrument(product, "main") != row.instrument(product, "main"):
                symbols.add(assignments[index - 1].instrument(product, "main").lower())
            for symbol in symbols:
                if day not in files[symbol]:
                    raise FileNotFoundError(f"{day}/{product} 缺少 {symbol} Bar")
    print(f"products={','.join(products)} requested={start_day}..{end_day} "
          f"actual={days[0]}..{days[-1]} days={len(days)} "
          f"contracts={len(selected)}", flush=True)

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
    for index, (day, row) in enumerate(zip(days, assignments)):
        for product in products:
            symbols = {row.instrument(product, "main").lower()}
            if index and assignments[index - 1].instrument(product, "main") != row.instrument(product, "main"):
                symbols.add(assignments[index - 1].instrument(product, "main").lower())
            for symbol in sorted(symbols):
                path = files[symbol][day]
                feed.add_bar_feather(path, FixedInstrumentBarParser(
                    instruments[symbol], timestamp=_timestamp_column(path),
                ))

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
        target_notional=target_notional,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
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
    try:
        result = runtime.run()
        trader = backend.engine.trader
        orders = trader.generate_orders_report()
        fills = trader.generate_fills_report()
        print(f"bars={result.replay_summary.bars} frames={strategy.synchronized_frames} "
              f"rebalances={strategy.rebalances} orders={len(orders)} fills={len(fills)}")
        if not strategy.rebalances:
            raise AssertionError("共同分钟 Bar 不足以完成回看窗口和调仓")
        if client.report_errors:
            raise AssertionError(f"模拟执行回报异常: {client.report_errors}")
        if require_fills and fills.empty:
            raise AssertionError("本次要求成交，但没有模拟成交")
        output = (report_dir or Path(__file__).resolve().parent / "results") / str(result.backend_result.run_id)
        output.mkdir(parents=True, exist_ok=True)
        orders.to_csv(output / "orders.csv")
        fills.to_csv(output / "fills.csv")
        trader.generate_positions_report().to_csv(output / "positions.csv")
        for venue_name in profiles:
            trader.generate_account_report(Venue(venue_name)).to_csv(output / f"account_{venue_name}.csv")
        summary = (asdict(result.backend_result) if is_dataclass(result.backend_result)
                   else {"backend_result": str(result.backend_result)})
        summary["strategy"] = {
            "products": products, "contracts": tuple(selected),
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
    finally:
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="CTP 多品种动态主力截面动量回测")
    parser.add_argument("--products", required=True, help="逗号分隔的品种，例如 RB,HC,I")
    parser.add_argument("--start-day", required=True, type=date.fromisoformat)
    parser.add_argument("--end-day", required=True, type=date.fromisoformat)
    parser.add_argument("--bars-dir", type=Path, help="覆盖 KLINE_DIR")
    parser.add_argument("--contract-struct", type=Path, help="覆盖 ROLE_DIR/fut_contract_data.feather")
    parser.add_argument("--fut-basic", type=Path, help="覆盖 ROLE_DIR/fut_basic.feather")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--target-notional", type=Decimal, default=Decimal("100000"))
    parser.add_argument("--starting-balance", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--commission-per-contract", type=Decimal, default=Decimal("1"))
    parser.add_argument("--margin-init", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--margin-maint", type=Decimal, default=Decimal("0.08"))
    parser.add_argument("--max-notional", type=Decimal, default=Decimal("1000000"))
    parser.add_argument("--log-level", choices=("ERROR", "WARNING", "INFO", "DEBUG"), default="WARNING")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--no-tearsheet", action="store_true")
    parser.add_argument("--require-fills", action="store_true")
    args = parser.parse_args()
    bars = args.bars_dir or (Path(os.environ["KLINE_DIR"]).expanduser()
                            if os.environ.get("KLINE_DIR") else None)
    basic = args.fut_basic or (Path(os.environ["ROLE_DIR"]).expanduser() / "fut_basic.feather"
                               if os.environ.get("ROLE_DIR") else None)
    roles = args.contract_struct or (Path(os.environ["ROLE_DIR"]).expanduser() / "fut_contract_data.feather"
                                    if os.environ.get("ROLE_DIR") else None)
    if bars is None or basic is None or roles is None:
        parser.error("请设置 KLINE_DIR 和 ROLE_DIR，或传入 --bars-dir、--contract-struct、--fut-basic")
    run_case(products=tuple(item.strip() for item in args.products.split(",")),
             start_day=args.start_day, end_day=args.end_day,
             bars_dir=bars, contract_struct=roles, fut_basic=basic, lookback=args.lookback,
             rebalance_interval=args.rebalance_interval, target_notional=args.target_notional,
             starting_balance=args.starting_balance,
             commission_per_contract=args.commission_per_contract,
             margin_init=args.margin_init, margin_maint=args.margin_maint,
             max_notional=args.max_notional, log_level=args.log_level,
             report_dir=args.report_dir, tearsheet=not args.no_tearsheet,
             require_fills=args.require_fills)


if __name__ == "__main__":
    main()
