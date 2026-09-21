#!/usr/bin/env python3
"""统一积木链的Binance在线EMA示例。

默认使用``market.stream.bn``接收已收盘Kline，并把目标发送到
``RecordingExecutionClient``，绝不会下单。只有显式传入``--enable-orders``才会
装配Nautilus Binance执行客户端；LIVE环境还必须再传``--confirm-live``。
"""

from __future__ import annotations

import argparse
import os
import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from bomber.adapters.binance import (
    BINANCE,
    BinanceAccountType,
    BinanceDataClientConfig,
    BinanceExecClientConfig,
    BinanceInstrumentProviderConfig,
    BinanceLiveDataClientFactory,
    BinanceLiveExecClientFactory,
)
from bomber.adapters.binance.common.enums import BinanceEnvironment
from bomber.config import LiveExecEngineConfig, LoggingConfig, TradingNodeConfig
from bomber.live.node import TradingNode
from bomber.model.identifiers import InstrumentId, TraderId

from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy
from market.basic.base import DataType, InstrumentMeta
from market.stream.bn import BNWSConfig, BNWSStreamDataFeed
from strategy import (
    BackendExecutionClient,
    DataBinding,
    ExecutionRoute,
    MarketReferencePriceStore,
    NautilusLiveExecutionBackend,
    NautilusTradingNodeDriver,
    NetTargetOrderPlanner,
    PositionManager,
    PreTradeRiskManager,
    RecordingExecutionClient,
    RiskLimits,
    RuntimeMode,
    UnifiedStrategyRunner,
)


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _environment(name: str) -> BinanceEnvironment:
    return BinanceEnvironment.DEMO if name == "demo" else BinanceEnvironment.LIVE


def _check_execution_authorization(environment: str, enabled: bool, confirm_live: bool) -> None:
    if not enabled:
        return
    if environment == "live" and not confirm_live:
        raise SystemExit("LIVE真实下单必须同时传入 --enable-orders --confirm-live")
    prefix = "BINANCE_DEMO" if environment == "demo" else "BINANCE"
    missing = [name for name in (f"{prefix}_API_KEY", f"{prefix}_API_SECRET") if not os.getenv(name)]
    if missing:
        raise SystemExit(f"启用下单前必须设置环境变量: {', '.join(missing)}")


def _node_initialized(node: TradingNode, instrument_id: InstrumentId) -> bool:
    initialized = getattr(node.portfolio, "initialized", False)
    if callable(initialized):
        initialized = initialized()
    return bool(initialized) and node.cache.instrument(instrument_id) is not None


def _account_positions(node: TradingNode, instrument_id: InstrumentId):
    return {instrument_id: Decimal(str(node.portfolio.net_position(instrument_id)))}


def _build_execution_client(args, instrument_id, positions, prices):
    if not args.enable_orders:
        client = RecordingExecutionClient("recording-only")
        return client, None

    environment = _environment(args.environment)
    provider = BinanceInstrumentProviderConfig(load_ids=frozenset([instrument_id]))
    node = TradingNode(
        config=TradingNodeConfig(
            trader_id=TraderId("EMA-LIVE-001"),
            logging=LoggingConfig(log_level="INFO"),
            exec_engine=LiveExecEngineConfig(
                reconciliation=True,
                graceful_shutdown_on_exception=True,
            ),
            # 该原生Data Client只服务TradingNode的合约加载、账户估值和对账；
            # EMA策略行情仍唯一来自外部BNWSStreamDataFeed。
            data_clients={
                BINANCE: BinanceDataClientConfig(
                    account_type=BinanceAccountType.USDT_FUTURES,
                    environment=environment,
                    instrument_provider=provider,
                ),
            },
            exec_clients={
                BINANCE: BinanceExecClientConfig(
                    account_type=BinanceAccountType.USDT_FUTURES,
                    environment=environment,
                    instrument_provider=provider,
                    max_retries=3,
                ),
            },
            timeout_connection=30.0,
            timeout_reconciliation=30.0,
            timeout_portfolio=30.0,
            timeout_disconnection=10.0,
            timeout_post_stop=5.0,
        ),
    )
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)
    driver = NautilusTradingNodeDriver(
        "binance-live",
        node,
        reconcile_callback=lambda: _account_positions(node, instrument_id),
        ready_callback=lambda: _node_initialized(node, instrument_id),
        startup_timeout=60.0,
    )
    backend = NautilusLiveExecutionBackend("binance-live", driver)
    limits = RiskLimits(
        max_order_quantity=Decimal(args.max_order_quantity),
        max_abs_position=Decimal(args.max_position),
        max_order_notional=Decimal(args.max_order_notional),
        max_abs_position_notional=Decimal(args.max_position_notional),
        max_market_age_ns=180 * 1_000_000_000,
    )
    client = BackendExecutionClient(
        backend.backend_id,
        NetTargetOrderPlanner(positions),
        backend,
        positions,
        risk_manager=PreTradeRiskManager(
            backend.backend_id,
            positions,
            prices,
            instrument_limits={instrument_id: limits},
        ),
    )
    return client, backend


def build_runner(args):
    _check_execution_authorization(args.environment, args.enable_orders, args.confirm_live)
    symbol = args.symbol.upper()
    instrument_id = InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
    default_ws = (
        "wss://demo-fstream.binance.com"
        if args.environment == "demo"
        else "wss://fstream.binance.com"
    )
    feed = BNWSStreamDataFeed(
        BNWSConfig(
            ws_base_url=args.ws_base_url or os.getenv("BN_WS_BASE_URL", default_ws),
            market_type="futures",
        ),
        source_id="BINANCE_EMA_KLINE",
    )
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=args.price_precision,
            size_precision=args.size_precision,
            price_increment=Decimal(args.price_increment),
            multiplier=Decimal(1),
            currency="USDT",
            exchange="BINANCE",
        ),
    )
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    client, backend = _build_execution_client(args, instrument_id, positions, prices)
    strategy = EmaCrossTargetStrategy(
        "ema-binance-live",
        EmaCrossConfig(
            fast_period=args.fast,
            slow_period=args.slow,
            long_quantity=Decimal(args.quantity),
            short_quantity=-Decimal(args.quantity),
            skip_single_price=False,
        ),
    )
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("binance-kline", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(
            DataBinding(
                "primary_bar",
                "binance-kline",
                instrument_id,
                DataType.BAR,
                args.interval.upper(),
            ),
        ),
        execution_routes=(
            ExecutionRoute("position", client.client_id, instrument_id),
        ),
    )
    return runner, strategy, client, backend, instrument_id


def main() -> None:
    parser = argparse.ArgumentParser(description="统一积木链Binance在线EMA策略")
    parser.add_argument("--environment", choices=("demo", "live"), default="demo")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--interval",
        choices=("1-MINUTE", "3-MINUTE", "5-MINUTE", "15-MINUTE", "30-MINUTE", "1-HOUR"),
        default="1-MINUTE",
    )
    parser.add_argument("--fast", type=int, default=3)
    parser.add_argument("--slow", type=int, default=5)
    parser.add_argument("--quantity", default="0.001")
    parser.add_argument("--price-precision", type=int, default=2)
    parser.add_argument("--size-precision", type=int, default=6)
    parser.add_argument("--price-increment", default="0.01")
    parser.add_argument("--ws-base-url")
    parser.add_argument("--timeout", type=float, default=0.0, help="0表示运行到Ctrl+C")
    parser.add_argument("--enable-orders", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--max-order-quantity", default="0.01")
    parser.add_argument("--max-position", default="0.02")
    parser.add_argument("--max-order-notional", default="2000")
    parser.add_argument("--max-position-notional", default="3000")
    args = parser.parse_args()

    runner, strategy, client, backend, instrument_id = build_runner(args)
    mode = "统一Live Backend真实执行" if args.enable_orders else "Recording-only，不下单"
    print(
        f"启动Binance {args.environment.upper()} EMA：instrument={instrument_id} "
        f"bar={args.interval.upper()} fast={args.fast} slow={args.slow} mode={mode}",
    )
    runner.start()
    started = time.monotonic()
    recorded = 0
    reported = 0
    try:
        while args.timeout <= 0 or time.monotonic() - started < args.timeout:
            if isinstance(client, RecordingExecutionClient):
                requests = client.requests
                for request in requests[recorded:]:
                    print(f"Recording目标请求: {request}")
                recorded = len(requests)
            elif backend is not None:
                reports = backend.reports
                for report in reports[reported:]:
                    print(f"执行回报: {report}")
                reported = len(reports)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("收到Ctrl+C，停止EMA在线策略")
    finally:
        runner.stop()
    print(
        f"Binance EMA已停止: bars_seen={strategy.bars_seen} "
        f"bars_used={strategy.bars_used} last_target={strategy.last_target}",
    )


if __name__ == "__main__":
    main()
