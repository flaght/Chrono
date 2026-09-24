"""第四类M4b：RB四角色信号与真实合约Bar的基础模拟回测装配。

这里使用已有CTP Basic Profile，不宣称已复现上期所平今/平昨、交易日结算；
真实Bar只用于撮合，DataHub复权价只用于信号。单日无穿越时可以零订单结束。
"""
from datetime import date
from decimal import Decimal
from pathlib import Path
import pandas as pd

from bomber.backtest.config import BacktestEngineConfig
from bomber.model import Venue
from bomber.model.identifiers import InstrumentId, TraderId


from market.replay.parsers.bar import (
    BarColumns,
    BinanceKlineParser,
    BinanceMarketType,
    MappedBarParser
)
from market.replay.base import FileReplayFeed
from market.basic.base import DataType, InstrumentMeta
from datahub import MinimalDataHub


from trader import (
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

from examples.role_cross.role_strategy import RoleCrossConfig, RoleCrossTargetStrategy
from examples.role_cross.local_input import load_role_research

ROLE_ROOT = Path("/workspace/worker/pj/neutron/tests/temp/role")
BASE_ROOT = Path(
    "/workspace/data/dev/kd/intelkit/records/temp")

DAY_NS = 86_400_000_000_000

DEFAULT_FUT_CONTRACT_DATA = ROLE_ROOT / "fut_contract_data.feather"
DEFAULT_FUT_ADJUSTMENT_FACTORS = ROLE_ROOT / "fut_adjustment_factors.feather"
DEFAULT_FUT_CONTRACT_STRUCTURES = ROLE_ROOT / "fut_contract_structures.feather"
DEFAULT_FUT_BASIC = ROLE_ROOT / "fut_basic.feather"


def _bar_path(root: Path, symbol: str, day: date) -> Path:
    matches = tuple(root.rglob(f"{symbol}_{day:%Y%m%d}.feather"))
    if len(matches) != 1:
        print(
            f"{day}/{symbol}需要唯一真实合约Bar文件，找到{len(matches)}个: {root}",
        )
        return None
    return matches[0]

def _utc_day_ns(value: object) -> int:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("fut_basic合约日期不能为空")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.tz_convert("UTC").value)

def create_instrument(instrument_row, profile):
    multiplier = Decimal(str(instrument_row["contMultNum"]))
    activation_ns = _utc_day_ns(instrument_row["listDate"])
    expiration_ns = _utc_day_ns(instrument_row["lastTradeDate"]) + DAY_NS
    price_increment = Decimal(str(instrument_row["minChgPriceNum"]))
    precision = max(0, -price_increment.normalize().as_tuple().exponent)
    instrument = profile.make_instrument(
        instrument_row['symbol'],
        underlying=instrument_row['code'],
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
    price_increment: Decimal,
    quantity: Decimal = Decimal(1),
    require_fills: bool = False,
) -> None:
    loaded = load_role_research(
        bars_dir=BASE_ROOT,
        contract_struct_path=DEFAULT_FUT_CONTRACT_DATA,
        factors_path=DEFAULT_FUT_ADJUSTMENT_FACTORS,
        fut_basic_path=DEFAULT_FUT_BASIC,
    )
    ends = dict(loaded.day_end_ns)
    days = tuple(day for day in sorted(ends) if start_day <= day <= end_day)
    assignments = tuple(loaded.store.assignment_at(ends[day]) for day in days)
    all_symbols = tuple(
        sorted({symbol for row in assignments for symbol in row.contracts.values()}))
    basic = pd.read_feather(DEFAULT_FUT_BASIC)
    required = {"symbol", "exchangeCD",
                "contMultNum", "listDate", "lastTradeDate"}
    # 上述为特殊处理

    parser = MappedBarParser(
            columns=BarColumns(
                symbol="symbol",
                exchange="exchange",
                timestamp="datetime",
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                value="value",
                open_interest="open_interest",
                vwap="vwap",
            ),
            bar_spec="1-MINUTE",
            timezone="Asia/Shanghai",
            exchange_aliases={"XSGE": "SHFE", "XZCE": "CZCE", "XDCE": "DCE"},
        )

    feed = FileReplayFeed("ROLE_CROSS_FORMAL_REPLAY")

    backend = NautilusSimExecutionBackend(
        "role-cross-sim",
        BacktestEngineConfig(trader_id=TraderId(
            "ROLE-CROSS-001"), run_analysis=True),
    )

    profile = CtpFuturesBasicProfile(
            starting_balance=Decimal("1000000"),
            commission_per_contract=Decimal(1),
            venue=Venue("SHFE"),
        )
    backend.add_profile(
        profile
    )
    instruments: dict[str, InstrumentId] = {}
    multipliers: dict[InstrumentId, Decimal] = {}

    for symbol in all_symbols:
        instrument_row= basic[basic['symbol'].isin([symbol])]
        if instrument_row.empty:
            raise ValueError(f"fut_basic缺少需要回放的合约: {symbol}")
        instrument, meta, multiplier = create_instrument(
            instrument_row.to_dict(orient="records")[0], profile)
        instruments[symbol] = instrument.id
        backend.add_instrument(instrument)
        feed.register_instrument(meta)
        print(instrument)
        multipliers[instrument.id] = multiplier


    for index, (day, assignment) in enumerate(zip(days, assignments)):
        required_symbols = dict.fromkeys(assignment.contracts.values())
        if index and assignments[index - 1].contracts["main"] != assignment.contracts["main"]:
            required_symbols[assignments[index - 1].contracts["main"]] = None
        for symbol in required_symbols:
            path = _bar_path(BASE_ROOT, symbol, day)
            if path is None:
                continue
            feed.add_bar_feather(
                path,
                parser
            )

    ### 主力合约解析
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
            f"signals={len(strategy.signal_events)} "
            f"orders={len(orders)} fills={len(fills)} "
            f"roll_phase={None if roll is None else roll.phase.value} "
            f"active_main={None if roll is None else roll.active.instrument_id}",
        )
        signal = strategy.signal
        print(
            f"M4b价差诊断: first={signal.first_spread} "
            f"min={signal.min_spread} max={signal.max_spread} "
            f"last={signal.previous_spread} "
            f"positive={signal.positive_frames} "
            f"negative={signal.negative_frames} zero={signal.zero_frames}",
        )
        print(f"M4b不可用事件: {strategy.unavailable_events}")
        print(f"M4b复权缺口: {loaded.store.factor_gaps}")
        print(f"M4b换约锚点: {loaded.store.factor_anchors}")
        if signal.last_processed_ns >= 0:
            last_frame = pd.Timestamp(signal.last_processed_ns, unit="ns", tz="UTC")
            print(f"M4b最后有效帧: {last_frame.tz_convert('Asia/Shanghai')}")
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
    run_case(
        start_day=date(2026, 1, 5),
        end_day=date(2026, 2, 26),
        price_increment=Decimal("0.01"),
    )


if __name__ == "__main__":
    main()
