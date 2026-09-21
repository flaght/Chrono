"""使用一天CTP Tick或Bar运行正式EMA撮合回测。"""

import argparse
from dataclasses import dataclass
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.model import BarType, Venue
from bomber.model.currencies import CNY
from bomber.model.enums import AssetClass
from bomber.model.identifiers import InstrumentId, Symbol, TraderId
from bomber.model.instruments import FuturesContract
from bomber.model.objects import Price, Quantity

from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import BarColumns, MappedBarParser
from market.replay.parsers.tick import CtpTickParser
from strategy import (
    CtpFuturesBasicProfile,
    DataBinding,
    ExecutionRoute,
    ExecutionReportType,
    MarketStreamBinding,
    MarketReferencePriceStore,
    NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend,
    NetTargetOrderPlanner,
    PositionManager,
    PreTradeRiskManager,
    RiskLimits,
    RuntimeMode,
    SimulationExecutionClient,
    UnifiedHistoricalRuntime,
    UnifiedStrategyRunner,
)
from strategy.bridge import (
    NautilusStrategyEventBridge,
    NautilusStrategyEventBridgeConfig,
)
from examples.single_ema.strategies.ema_cross import EmaCrossConfig, EmaCrossTargetStrategy

BAR_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728/"
    "rb2704_20260728.feather",
)
TICK_PATH = Path(
    "/workspace/data/fut_tick/7050707549_-/2026/202607/20260728/"
    "rb2609_20260728.csv",
)
BAR_ID = InstrumentId.from_str("rb2704.SHFE")
TICK_ID = InstrumentId.from_str("rb2609.SHFE")


@dataclass(frozen=True)
class _ExpectedResult:
    loaded: int
    bars_used: int
    orders: int
    fills: int


_EXPECTED_RESULTS = {
    "bar": _ExpectedResult(loaded=351, bars_used=7, orders=1, fills=1),
    "tick": _ExpectedResult(loaded=42_530, bars_used=112, orders=45, fills=45),
}

_RB_RISK_LIMITS = RiskLimits(
    max_order_quantity=Decimal(2),
    max_abs_position=Decimal(1),
    max_order_notional=Decimal("50000"),
    max_abs_position_notional=Decimal("50000"),
    max_market_age_ns=120 * 1_000_000_000,
    contract_multiplier=Decimal(10),
)


def _ns(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000_000_000)

def _future(instrument_id: InstrumentId, expiration: str) -> FuturesContract:
    return FuturesContract(
        instrument_id=instrument_id,
        raw_symbol=Symbol(instrument_id.symbol.value),
        asset_class=AssetClass.COMMODITY,
        currency=CNY,
        price_precision=0,
        price_increment=Price.from_str("1"),
        multiplier=Quantity.from_int(10),
        lot_size=Quantity.from_int(1),
        underlying="rb",
        activation_ns=_ns("2025-01-01"),
        expiration_ns=_ns(expiration),
        margin_init=Decimal("0.10"),
        margin_maint=Decimal("0.08"),
        exchange="SHFE",
        ts_event=0,
        ts_init=0,
    )

def _meta(instrument_id: InstrumentId) -> InstrumentMeta:
    return InstrumentMeta(
        instrument_id=instrument_id,
        price_precision=0,
        size_precision=0,
        price_increment=Decimal(1),
        multiplier=Decimal(10),
        currency="CNY",
        exchange="SHFE",
    )

def _configure_bar_feed() -> FileReplayFeed:
    feed = FileReplayFeed("CTP_BAR_BACKTEST_FEED")
    feed.register_instrument(_meta(BAR_ID))
    feed.add_bar_feather(
        BAR_PATH,
        MappedBarParser(
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
            exchange_aliases={"XSGE": "SHFE"},
        ),
    )
    return feed


def _configure_tick_feed() -> FileReplayFeed:
    feed = FileReplayFeed("CTP_TICK_BACKTEST_FEED")
    feed.register_instrument(_meta(TICK_ID))
    feed.add_tick_csv(
        TICK_PATH,
        CtpTickParser(
            exchange="SHFE",
            night_session_action_day="20260727",
        ),
    )
    return feed

def _build_runtime(case: str):
    if case == "bar":
        instrument_id = BAR_ID
        instrument = _future(BAR_ID, "2027-05-01")
        feed = _configure_bar_feed()
        data_bindings = (
            DataBinding("primary_bar", "ctp-replay", BAR_ID, DataType.BAR, "1-MINUTE"),
        )
        market_bindings = (
            MarketStreamBinding(BAR_ID, DataType.BAR, "1-MINUTE"),
        )
    else:
        instrument_id = TICK_ID
        instrument = _future(TICK_ID, "2026-10-01")
        feed = _configure_tick_feed()
        # 原始Tick既推进Nautilus唯一模拟时钟，也让Runner明确知道策略依赖的
        # Feed；EMA真正消费的1分钟Bar由下方EventBridge转发。
        data_bindings = (
            DataBinding("raw_quote", "ctp-replay", TICK_ID, DataType.QUOTE_TICK),
            DataBinding("raw_trade", "ctp-replay", TICK_ID, DataType.TRADE_TICK),
        )
        market_bindings = (
            MarketStreamBinding(TICK_ID, DataType.QUOTE_TICK),
            MarketStreamBinding(TICK_ID, DataType.TRADE_TICK),
        )

    backend = NautilusSimExecutionBackend(
        f"ctp-{case}-sim",
        BacktestEngineConfig(trader_id=TraderId("BACKTESTER-001"), run_analysis=True),
    )
    backend.add_profile(
        CtpFuturesBasicProfile(
            starting_balance=Decimal("1000000"),
            commission_per_contract=Decimal(1),
            venue=Venue("SHFE"),
        ),
    )
    backend.add_instrument(instrument)
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
            instrument_limits={instrument_id: _RB_RISK_LIMITS},
        ),
    )
    target_strategy = EmaCrossTargetStrategy(
        f"ema-{case}",
        EmaCrossConfig(fast_period=3, slow_period=5),
    )
    event_bridge = None
    if case == "tick":
        event_bridge = NautilusStrategyEventBridge(
            NautilusStrategyEventBridgeConfig(
                instrument_id=instrument_id,
                bar_type=BarType.from_str(
                    f"{instrument_id}-1-MINUTE-LAST-INTERNAL",
                ),
                data_key="primary_bar",
            ),
            target_strategy,
        )
        backend.add_strategy(event_bridge)

    runner = UnifiedStrategyRunner(
        RuntimeMode.HISTORICAL,
        position_manager=positions,
    )
    runner.add_market_observer(prices)
    runner.add_data_feed("ctp-replay", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        target_strategy,
        data_bindings=data_bindings,
        execution_routes=(
            ExecutionRoute("position", backend.backend_id, instrument_id),
        ),
    )
    market_adapter = NautilusMarketFeedAdapter(
        f"ctp-{case}-clock",
        feed,
        backend,
        market_bindings,
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime(
        f"ctp-{case}-runtime",
        runner,
        market_adapter,
    )
    return runtime, target_strategy, backend, event_bridge, client

def run_case(case: str) -> None:
    if case not in _EXPECTED_RESULTS:
        raise ValueError(f"不支持的CTP回测数据源: {case}")
    instrument_id = BAR_ID if case == "bar" else TICK_ID
    runtime, target_strategy, backend, event_bridge, client = _build_runtime(case)
    try:
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        positions = backend.engine.trader.generate_positions_report()
        loaded = result.replay_summary.total
        print(f"[{case}] loaded={loaded:,} bars_used={target_strategy.bars_used:,}")
        print(
            f"[{case}] last_target={target_strategy.last_target} orders={len(orders)} "
            f"fills={len(fills)} positions={len(positions)}",
        )
        print(f"[{case}] result={result.backend_result}")
        reference = client.risk_manager.price_store.get(instrument_id)
        if reference is None:
            raise AssertionError(f"{case}风控参考价没有收到行情")
        print(
            f"[{case}] risk_price={reference.price} risk_ts={reference.ts_event} "
            f"contract_multiplier={_RB_RISK_LIMITS.contract_multiplier}",
        )
        if target_strategy.bars_used < 5:
            raise AssertionError(f"{case}没有产生足够的EMA输入Bar")
        if target_strategy.last_target is None:
            raise AssertionError(f"{case}策略没有产生目标仓位")
        if orders.empty or fills.empty:
            raise AssertionError(f"{case}正式回测没有产生订单或成交")
        expected = _EXPECTED_RESULTS[case]
        actual = _ExpectedResult(
            loaded=loaded,
            bars_used=target_strategy.bars_used,
            orders=len(orders),
            fills=len(fills),
        )
        if actual != expected:
            raise AssertionError(
                f"{case}回测结果偏离固定基准: actual={actual} expected={expected}",
            )
        if client.report_errors:
            raise AssertionError(f"{case}执行回报状态异常: {client.report_errors}")
        filled_reports = tuple(
            report
            for report in backend.reports
            if report.report_type is ExecutionReportType.FILLED
        )
        if len(filled_reports) != expected.orders:
            raise AssertionError(
                f"{case}统一成交回报数量异常: {len(filled_reports)} != {expected.orders}",
            )
        equal_timestamp_fills = 0
        for report in filled_reports:
            signal_ts = int(report.metadata["signal_ts"])
            if report.ts_event < signal_ts:
                raise AssertionError(
                    f"{case}检测到成交早于信号: fill={report.ts_event} signal={signal_ts}",
                )
            if report.ts_event == signal_ts:
                equal_timestamp_fills += 1
                # 外部Feather Bar一根一根推进，同时间戳成交才意味着同Bar撮合。
                # 内部Tick聚合的Bar和下一份Tick可能共享CTP毫秒时间戳；
                # 时间戳相等不能单独证明是否为同一个行情事件。
                if case == "bar":
                    raise AssertionError("bar检测到同Bar成交")
        if equal_timestamp_fills:
            print(
                f"[{case}] 与信号同时间戳的成交={equal_timestamp_fills}；"
                "CTP Tick需按行情事件顺序进一步审计，不视为已证明N→N+1",
            )
        if client.position_manager.working_quantity(backend.backend_id, instrument_id) != 0:
            raise AssertionError(f"{case}回测结束后仍存在未释放在途数量")
        if client.position_manager.account_position(backend.backend_id, instrument_id) != target_strategy.last_target:
            raise AssertionError(f"{case}账户仓位与EMA最终目标不一致")
        if case == "tick" and (event_bridge is None or event_bridge.bars_forwarded < 5):
            raise AssertionError("Tick没有通过只读EventBridge形成足够的内部Bar")
    finally:
        runtime.stop()

def main() -> None:
    parser = argparse.ArgumentParser(description="单标的EMA CTP正式回测")
    parser.add_argument(
        "--source",
        choices=("bar", "tick", "all"),
        default="all",
        help="选择Feather Bar、CSV Tick或分别运行两者",
    )
    args = parser.parse_args()
    if args.source == "all":
        script = str(Path(__file__).resolve())
        for case in ("bar", "tick"):
            subprocess.run(
                [sys.executable, script, "--source", case],
                check=True,
            )
        return
    print(f"\n=== CTP EMA {args.source.upper()}正式回测 ===", flush=True)
    run_case(args.source)


if __name__ == "__main__":
    main()
