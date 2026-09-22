"""第四类M4b：RB四角色信号与真实合约Bar的基础模拟回测装配。

这里使用已有CTP Basic Profile，不宣称已复现上期所平今/平昨、交易日结算；
真实Bar只用于撮合，DataHub复权价只用于信号。单日无穿越时可以零订单结束。
"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
from bomber.backtest.config import BacktestEngineConfig
from bomber.model import Venue
from bomber.model.identifiers import InstrumentId, TraderId

from datahub import MinimalDataHub
from examples.role_cross.local_input import load_rb_role_research
from examples.role_cross.role_strategy import RoleCrossConfig, RoleCrossTargetStrategy
from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import FixedInstrumentBarParser
from strategy import (
    ContractAssignment,
    CtpFuturesBasicProfile,
    DataBinding,
    DynamicExecutionRoute,
    MarketReferencePriceStore,
    MarketStreamBinding,
    NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend,
    NetTargetOrderPlanner,
    PositionManager,
    PreTradeRiskManager,
    RiskLimits,
    RuntimeMode,
    ScheduledContractResolver,
    SimulationExecutionClient,
    UnifiedHistoricalRuntime,
    UnifiedStrategyRunner,
)


ROLE_ROOT = Path("/workspace/worker/pj/neutron/tests/temp/role")
DAY_NS = 86_400_000_000_000


def _utc_day_ns(value: object) -> int:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("fut_basic合约日期不能为空")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.tz_convert("UTC").value)


def _bar_path(root: Path, symbol: str, day: date) -> Path:
    matches = tuple(root.rglob(f"{symbol}_{day:%Y%m%d}.feather"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{day}/{symbol}需要唯一真实合约Bar文件，找到{len(matches)}个: {root}",
        )
    return matches[0]


def _timestamp_column(path: Path) -> str:
    import pyarrow as pa

    for name in ("datetime", "timestamp"):
        try:
            pd.read_feather(path, columns=[name])
        except (KeyError, ValueError, pa.ArrowInvalid):
            continue
        return name
    raise ValueError(f"Bar文件缺少datetime/timestamp列: {path}")


def _instrument(
    basic: pd.DataFrame,
    profile: CtpFuturesBasicProfile,
    symbol: str,
    price_increment: Decimal,
):
    selected = basic.loc[basic["symbol"].astype(str).str.strip().str.lower() == symbol]
    if len(selected) != 1:
        raise ValueError(f"fut_basic需要唯一合约记录: {symbol}, 实际={len(selected)}")
    row = selected.iloc[0]
    if str(row["exchangeCD"]).upper() not in {"XSGE", "SHFE"}:
        raise ValueError(f"RB合约交易所应为SHFE: {symbol}/{row['exchangeCD']}")
    multiplier = Decimal(str(row["contMultNum"]))
    if not multiplier.is_finite() or multiplier <= 0:
        raise ValueError(f"合约乘数无效: {symbol}")
    activation_ns = _utc_day_ns(row["listDate"])
    expiration_ns = _utc_day_ns(row["lastTradeDate"]) + DAY_NS
    precision = max(0, -price_increment.normalize().as_tuple().exponent)
    instrument = profile.make_instrument(
        symbol,
        underlying="rb",
        price_precision=precision,
        price_increment=price_increment,
        multiplier=multiplier,
        activation_ns=activation_ns,
        expiration_ns=expiration_ns,
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.08"),
    )
    meta = InstrumentMeta(
        instrument.id,
        price_precision=precision,
        size_precision=0,
        price_increment=price_increment,
        multiplier=multiplier,
        currency="CNY",
        exchange="SHFE",
    )
    return instrument, meta, multiplier


def run_case(
    *,
    start_day: date,
    end_day: date,
    bars_dir: Path,
    contract_struct: Path,
    factors: Path,
    fut_basic: Path,
    price_increment: Decimal,
    quantity: Decimal = Decimal(1),
    require_fills: bool = False,
) -> None:
    if end_day < start_day:
        raise ValueError("结束交易日不能早于开始交易日")
    if not price_increment.is_finite() or price_increment <= 0:
        raise ValueError("价格步长必须为正且有限")
    loaded = load_rb_role_research(
        bars_dir=bars_dir,
        contract_struct_path=contract_struct,
        factors_path=factors,
        fut_basic_path=fut_basic,
    )
    for day, role, reason in loaded.store.factor_gaps:
        print(f"M4b研究价缺口: day={day} role={role} reason={reason}；跳过受影响的策略决策")
    for day, role, source_day, anchor_day in loaded.store.factor_anchors:
        print(f"M4b换约锚点: day={day} role={role} source={source_day} anchor={anchor_day}")
    ends = dict(loaded.day_end_ns)
    days = tuple(day for day in sorted(ends) if start_day <= day <= end_day)
    if not days:
        raise ValueError("所选日期没有可用的前日角色表和RB Bar")
    assignments = tuple(loaded.store.assignment_at(ends[day]) for day in days)
    all_symbols = tuple(sorted({symbol for row in assignments for symbol in row.contracts.values()}))
    basic = pd.read_feather(fut_basic)
    required = {"symbol", "exchangeCD", "contMultNum", "listDate", "lastTradeDate"}
    if required - set(basic.columns):
        raise ValueError(f"fut_basic缺少正式回测元数据: {sorted(required - set(basic.columns))}")

    feed = FileReplayFeed("ROLE_CROSS_FORMAL_REPLAY")
    backend = NautilusSimExecutionBackend(
        "role-cross-sim",
        BacktestEngineConfig(trader_id=TraderId("ROLE-CROSS-001"), run_analysis=True),
    )
    profile = CtpFuturesBasicProfile(
        starting_balance=Decimal("1000000"),
        commission_per_contract=Decimal(1),
        venue=Venue("SHFE"),
    )
    backend.add_profile(profile)
    instruments: dict[str, InstrumentId] = {}
    multipliers: dict[InstrumentId, Decimal] = {}
    for symbol in all_symbols:
        instrument, meta, multiplier = _instrument(basic, profile, symbol, price_increment)
        instruments[symbol] = instrument.id
        multipliers[instrument.id] = multiplier
        backend.add_instrument(instrument)
        feed.register_instrument(meta)
    for index, (day, assignment) in enumerate(zip(days, assignments)):
        required_symbols = dict.fromkeys(assignment.contracts.values())
        if index and assignments[index - 1].contracts["main"] != assignment.contracts["main"]:
            # 新主力生效后旧主力不一定仍在四角色中，但平旧仓仍须有真实Bar撮合。
            required_symbols[assignments[index - 1].contracts["main"]] = None
        for symbol in required_symbols:
            path = _bar_path(bars_dir, symbol, day)
            feed.add_bar_feather(
                path,
                FixedInstrumentBarParser(
                    instruments[symbol], timestamp=_timestamp_column(path),
                ),
            )

    resolver = ScheduledContractResolver(tuple(
        ContractAssignment(
            "rb_main", instruments[row.contracts["main"]],
            row.effective_ns, row.available_ns, revision,
        )
        for revision, row in enumerate(assignments, 1)
    ))
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    client = SimulationExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id,
            positions,
            prices,
            instrument_limits={
                item: RiskLimits(
                    max_order_quantity=Decimal(2),
                    max_abs_position=Decimal(1),
                    max_order_notional=Decimal("100000"),
                    max_abs_position_notional=Decimal("100000"),
                    max_market_age_ns=120 * 1_000_000_000,
                    contract_multiplier=multipliers[item],
                )
                for item in multipliers
            },
        ),
    )
    strategy = RoleCrossTargetStrategy(
        "rb-role-cross-formal",
        MinimalDataHub(loaded.store),
        RoleCrossConfig(quantity=quantity),
    )
    adapter = NautilusMarketFeedAdapter(
        "role-cross-sim-clock", feed, backend,
        tuple(
            MarketStreamBinding(item, DataType.BAR, "1-MINUTE")
            for item in instruments.values()
        ),
        manage_lifecycle=False,
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("rb-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "rb-bars", item, DataType.BAR, "1-MINUTE")
            for item in instruments.values()
        ),
        execution_routes=(
            DynamicExecutionRoute("rb_main", backend.backend_id, resolver),
        ),
    )
    runtime = UnifiedHistoricalRuntime("role-cross-formal-runtime", runner, adapter)
    try:
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        roll = runner.roll_coordinator.state(strategy.strategy_id, "rb_main")
        print(
            f"M4b正式基础回测: days={len(days)} bars={result.replay_summary.bars} "
            f"complete_frames={strategy.complete_frames} "
            f"factor_gaps={len(loaded.store.factor_gaps)} "
            f"unavailable_events={strategy.unavailable_events} "
            f"signals={len(strategy.signal_events)} "
            f"orders={len(orders)} fills={len(fills)} "
            f"roll_phase={None if roll is None else roll.phase.value} "
            f"active_main={None if roll is None else roll.active.instrument_id}",
        )
        if strategy.complete_frames == 0:
            raise AssertionError("没有四角色同分钟完整帧，不能验收正式回测")
        if client.report_errors:
            raise AssertionError(f"模拟执行回报异常: {client.report_errors}")
        if require_fills and fills.empty:
            raise AssertionError("本次要求验收成交，但样本没有产生模拟成交")
        if not strategy.signal_events:
            print("本样本没有穿越信号；行情与模拟时钟已验证，订单/成交仍待跨零样本。")
        elif orders.empty:
            raise AssertionError("已有穿越信号却没有生成模拟订单")
        elif fills.empty:
            print("已有订单但未成交；请检查信号是否发生在最后一根Bar及撮合配置。")
    finally:
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="第四类RB角色穿越正式基础模拟回测")
    parser.add_argument("--start-day", type=date.fromisoformat, required=True)
    parser.add_argument("--end-day", type=date.fromisoformat, required=True)
    parser.add_argument("--bars-dir", type=Path, required=True,
                        help="只指向本次需要的有限日期目录，避免递归加载整个历史库")
    parser.add_argument("--fut-contract", type=Path, default=ROLE_ROOT / "fut_contract_data.feather")
    parser.add_argument("--factors", type=Path, default=ROLE_ROOT / "fut_adjustment_factors.feather")
    parser.add_argument("--fut-basic", type=Path, default=ROLE_ROOT / "fut_basic.feather")
    parser.add_argument("--price-increment", type=Decimal, required=True)
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--require-fills", action="store_true")
    args = parser.parse_args()
    run_case(
        start_day=args.start_day, end_day=args.end_day, bars_dir=args.bars_dir,
        contract_struct=args.fut_contract, factors=args.factors,
        fut_basic=args.fut_basic, price_increment=args.price_increment,
        quantity=args.quantity, require_fills=args.require_fills,
    )


if __name__ == "__main__":
    main()
