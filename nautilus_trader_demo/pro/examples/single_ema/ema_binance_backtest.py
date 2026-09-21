#!/usr/bin/env python3
"""Binance USDT永续一分钟Bar的统一EMA正式回测。"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.model.currencies import BTC
from bomber.model.identifiers import TraderId

from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy
from market.basic.base import DataType, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import BinanceKlineParser, BinanceMarketType
from strategy import (
    BinanceUsdtFuturesProfile,
    DataBinding,
    ExecutionReportType,
    ExecutionRoute,
    MarketReferencePriceStore,
    MarketStreamBinding,
    NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend,
    NetTargetOrderPlanner,
    PositionManager,
    PreTradeRiskManager,
    RuntimeMode,
    SimulationExecutionClient,
    UnifiedHistoricalRuntime,
    UnifiedStrategyRunner,
)


DEFAULT_BAR_PATH = Path(
    "/workspace/data/dev/kd/intelkit/records/raw_data/binance_data/"
    "futures/um/klines/1m/BTCUSDT/2023-09-02.csv",
)


def build_runtime(path: Path, *, fast: int, slow: int, quantity: Decimal):
    profile = BinanceUsdtFuturesProfile(
        starting_balance=Decimal("100000"),
        leverage=Decimal("10"),
        maintenance_margin_rate=Decimal("0.005"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
    )
    instrument = profile.make_instrument(
        "BTCUSDT",
        price_precision=2,
        size_precision=6,
        price_increment=Decimal("0.01"),
        size_increment=Decimal("0.000001"),
        base_currency=BTC,
        min_quantity=Decimal("0.000001"),
        min_notional=Decimal("5"),
    )
    instrument_id = instrument.id
    feed = FileReplayFeed("BINANCE_EMA_REPLAY")
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=2,
            size_precision=6,
            price_increment=Decimal("0.01"),
            multiplier=Decimal(1),
            currency="USDT",
            exchange="BINANCE",
        ),
    )
    feed.add_bar_csv(
        path,
        BinanceKlineParser(
            symbol="BTCUSDT",
            market_type=BinanceMarketType.FUTURES,
            interval="1m",
            include_factors=False,
        ),
    )

    backend = NautilusSimExecutionBackend(
        "binance-ema-sim",
        BacktestEngineConfig(
            trader_id=TraderId("BINANCE-EMA-001"),
            run_analysis=True,
        ),
    )
    backend.add_profile(profile)
    backend.add_instrument(instrument)
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    client = SimulationExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(backend.backend_id, positions, prices),
    )
    ema = EmaCrossTargetStrategy(
        "ema-binance-offline",
        EmaCrossConfig(
            fast_period=fast,
            slow_period=slow,
            long_quantity=quantity,
            short_quantity=-quantity,
            # Binance存在合法的单价K线；正式EMA默认仍应连续消费时间序列。
            skip_single_price=False,
        ),
    )
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("binance-replay", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        ema,
        data_bindings=(
            DataBinding(
                "primary_bar",
                "binance-replay",
                instrument_id,
                DataType.BAR,
                "1-MINUTE",
            ),
        ),
        execution_routes=(
            ExecutionRoute("position", backend.backend_id, instrument_id),
        ),
    )
    adapter = NautilusMarketFeedAdapter(
        "binance-ema-clock",
        feed,
        backend,
        (MarketStreamBinding(instrument_id, DataType.BAR, "1-MINUTE"),),
        manage_lifecycle=False,
    )
    runtime = UnifiedHistoricalRuntime("binance-ema-runtime", runner, adapter)
    return runtime, ema, client, backend, positions, instrument_id


def run_case(
    path: Path = DEFAULT_BAR_PATH,
    *,
    fast: int = 3,
    slow: int = 5,
    quantity: Decimal = Decimal("0.001"),
) -> None:
    runtime, ema, client, backend, positions, instrument_id = build_runtime(
        path,
        fast=fast,
        slow=slow,
        quantity=quantity,
    )
    try:
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        loaded = result.replay_summary.total
        filled_reports = tuple(
            report
            for report in backend.reports
            if report.report_type is ExecutionReportType.FILLED
        )
        print(
            f"[binance] loaded={loaded:,} bars_used={ema.bars_used:,} "
            f"last_target={ema.last_target} orders={len(orders)} fills={len(fills)}",
        )
        print(f"[binance] result={result.backend_result}")

        # 固定文件应完整读取全天1,440根一分钟Bar。
        if path == DEFAULT_BAR_PATH and loaded != 1_440:
            raise AssertionError(f"Binance固定基准应为1,440根Bar，实际{loaded}")
        if ema.bars_used != loaded:
            raise AssertionError("Binance EMA没有连续消费全部标准Bar")
        if ema.last_target is None or orders.empty or fills.empty:
            raise AssertionError("Binance EMA没有形成完整目标、订单和成交")
        if len(orders) != len(fills) or len(fills) != len(filled_reports):
            raise AssertionError("Binance订单、原生成交和统一成交回报数量不一致")
        if client.report_errors:
            raise AssertionError(f"Binance执行回报状态异常: {client.report_errors}")
        if positions.working_quantity(backend.backend_id, instrument_id) != 0:
            raise AssertionError("Binance回测结束后仍存在未释放在途数量")
        if positions.account_position(backend.backend_id, instrument_id) != ema.last_target:
            raise AssertionError("Binance账户仓位与EMA最终目标不一致")
        for report in filled_reports:
            signal_ts = int(report.metadata["signal_ts"])
            if report.ts_event <= signal_ts:
                raise AssertionError("检测到同Bar成交或前视：成交时间没有晚于信号时间")
        print("Binance EMA离线正式回测通过")
    finally:
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance单标的EMA正式回测")
    parser.add_argument("--path", type=Path, default=DEFAULT_BAR_PATH)
    parser.add_argument("--fast", type=int, default=3)
    parser.add_argument("--slow", type=int, default=5)
    parser.add_argument("--quantity", default="0.001")
    args = parser.parse_args()
    run_case(
        args.path,
        fast=args.fast,
        slow=args.slow,
        quantity=Decimal(args.quantity),
    )


if __name__ == "__main__":
    main()
