#!/usr/bin/env python3
"""P3 Binance DEMO账户只读探针；不会调用arm_demo或提交订单。"""

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

from strategy import (
    ControlledLiveExecutionClient,
    MarketReferencePriceStore,
    NautilusLiveExecutionBackend,
    NautilusReportedAccountReader,
    NautilusTradingNodeDriver,
    NetTargetOrderPlanner,
    PositionManager,
    PreTradeRiskManager,
    RiskLimits,
)
from strategy.execution.live.binance_orders import (
    BinanceHttpActiveOrderReader,
    BinanceNativeOpenOrdersBinding,
)


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _positions(node: TradingNode):
    """仅在节点完成原生启动对账后读取该DEMO账户的全部合约净仓。"""
    account = node.portfolio.account(BINANCE)
    if account is None:
        raise RuntimeError("Binance DEMO账户未建立")
    positions = {}
    for position in node.cache.positions_open(venue=BINANCE, account_id=account.id):
        instrument_id = position.instrument_id
        positions[instrument_id] = (
            positions.get(instrument_id, Decimal(0))
            + Decimal(str(position.signed_decimal_qty()))
        )
    return positions


def build_readonly_client(symbol: str):
    instrument_id = InstrumentId.from_str(f"{symbol.upper()}-PERP.BINANCE")
    # 仓位对账不能只加载BTC一个合约，否则其他持仓可能因合约未知而被遗漏。
    provider = BinanceInstrumentProviderConfig(load_all=True)
    node = TradingNode(config=TradingNodeConfig(
        trader_id=TraderId("P3-DEMO-READONLY-001"),
        logging=LoggingConfig(log_level="INFO"),
        exec_engine=LiveExecEngineConfig(
            reconciliation=True,
            graceful_shutdown_on_exception=True,
        ),
        data_clients={BINANCE: BinanceDataClientConfig(
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.DEMO,
            instrument_provider=provider,
        )},
        exec_clients={BINANCE: BinanceExecClientConfig(
            account_type=BinanceAccountType.USDT_FUTURES,
            environment=BinanceEnvironment.DEMO,
            instrument_provider=provider,
            max_retries=3,
        )},
        timeout_connection=30.0,
        timeout_reconciliation=30.0,
        timeout_portfolio=30.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    ))
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    open_orders_binding = BinanceNativeOpenOrdersBinding()

    class CapturingBinanceExecFactory(BinanceLiveExecClientFactory):
        """仅捕获本节点原生HTTP查询端点，不改变原生执行逻辑。"""

        @staticmethod
        def create(loop, name, config, msgbus, cache, clock):
            native_client = BinanceLiveExecClientFactory.create(
                loop, name, config, msgbus, cache, clock,
            )
            open_orders_binding.capture(native_client)
            return native_client

    node.add_exec_client_factory(BINANCE, CapturingBinanceExecFactory)
    reader = NautilusReportedAccountReader(
        "binance-demo", "binance-demo", lambda: node.portfolio.account(BINANCE),
        lambda native_id: driver.query_account(native_id),
        required_info_keys=("total_margin_balance", "available_balance"),
        timeout_seconds=15.0,
    )
    orders_reader = BinanceHttpActiveOrderReader(
        "binance-demo", "binance-demo", node.get_event_loop,
        open_orders_binding.query_open_orders,
        timeout_seconds=15.0,
    )
    driver = NautilusTradingNodeDriver(
        "binance-demo", node,
        reconcile_callback=lambda: _positions(node),
        account_state_callback=reader.read,
        active_orders_callback=orders_reader.read,
        ready_callback=lambda: (
            bool(node.portfolio.initialized)
            and node.portfolio.account(BINANCE) is not None
            and node.cache.instrument(instrument_id) is not None
        ),
        startup_timeout=60.0,
    )
    backend = NautilusLiveExecutionBackend("binance-demo", driver)
    positions = PositionManager()
    risk = PreTradeRiskManager(
        "binance-demo", positions, MarketReferencePriceStore(),
        default_limits=RiskLimits(max_order_quantity=Decimal("0.001")),
    )
    client = ControlledLiveExecutionClient(
        "binance-demo", NetTargetOrderPlanner(positions), backend, positions, risk,
        account_id="binance-demo",
        # 本探针永远只读；即使误调用arm_demo也无法打开下单闸门。
        demo_environment_check=lambda: False,
    )
    return client, driver


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance DEMO账户只读权威查询")
    parser.add_argument("--connect-demo", action="store_true", help="明确允许连接DEMO账户")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat", type=float, default=10.0)
    args = parser.parse_args()
    if not args.connect_demo:
        raise SystemExit("只读连接也需显式传入 --connect-demo；不会下单")
    if args.seconds <= 0 or args.heartbeat <= 0:
        raise SystemExit("--seconds和--heartbeat必须大于零")
    required = ("BINANCE_DEMO_API_KEY", "BINANCE_DEMO_API_SECRET")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"缺少DEMO凭据: {', '.join(missing)}")

    client, driver = build_readonly_client(args.symbol)
    print("P3只读探针：连接Binance DEMO账户，查询仓位及资金；不提交订单")
    client.start()
    try:
        state = client.account_state
        print(f"P3只读初始对账: revision={state.revision} currencies={tuple(state.balances)}")
        orders = driver.reconcile_active_orders()
        print(f"P4-BN2只读活动订单: revision={orders.revision} count={len(orders.orders)}")
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            time.sleep(min(args.heartbeat, max(0, deadline - time.monotonic())))
            if not driver.node.is_running():
                client.mark_disconnected("trading_node_stopped")
                raise RuntimeError("TradingNode已停止；只读探针闭闸")
            if time.monotonic() >= deadline:
                break
            state = client.refresh_account_state()
            print(f"P3只读资金心跳: revision={state.revision} currencies={tuple(state.balances)}")
    finally:
        client.stop()
    print("P3只读探针结束：未授权、未提交订单")


if __name__ == "__main__":
    main()
