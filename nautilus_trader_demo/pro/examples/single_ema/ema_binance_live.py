#!/usr/bin/env python3
"""Binance U本位期货 EMA：默认Recording，显式授权后仅允许DEMO下单。"""
from __future__ import annotations

import argparse
import os
import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from bomber.model.identifiers import InstrumentId
from examples.single_ema.binance_demo_readonly import build_readonly_client
from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy
from market.basic.base import DataType, InstrumentMeta
from market.stream.bn import BNWSConfig, BNWSStreamDataFeed
from market.stream.health import StreamHealthConfig
from trader import (DataBinding, ExecutionRoute, MarketReferencePriceStore,
                      PreTradeRiskManager, RecordingExecutionClient, RiskLimits,
                      RuntimeMode, UnifiedStrategyRunner)

load_dotenv(Path(__file__).resolve().parents[2] / '.env')


def build_runner(args):
    if args.enable_orders:
        if args.environment != 'demo' or not args.confirm_demo:
            raise SystemExit('仅允许DEMO报单，须同时指定 --enable-orders --confirm-demo')
        missing = [name for name in ('BINANCE_DEMO_API_KEY', 'BINANCE_DEMO_API_SECRET')
                   if not os.getenv(name)]
        if missing:
            raise SystemExit('缺少DEMO凭据: ' + ', '.join(missing))
    elif args.confirm_demo:
        raise SystemExit('--confirm-demo 只能和 --enable-orders 一起使用')
    instrument_id = InstrumentId.from_str(f'{args.symbol.upper()}-PERP.BINANCE')
    # 使用已验证的U本位期货公开行情；DEMO凭据仅交给TradingNode。
    feed = BNWSStreamDataFeed(
        BNWSConfig(ws_base_url=args.ws_base_url or 'wss://fstream.binance.com/market',
                   market_type='futures'),
        source_id='BINANCE_EMA_KLINE',
        health_config=StreamHealthConfig(startup_grace_seconds=90,
                                         stale_after_seconds=90),
    )
    feed.register_instrument(InstrumentMeta(
        instrument_id=instrument_id, price_precision=args.price_precision,
        size_precision=args.size_precision,
        price_increment=Decimal(args.price_increment), multiplier=Decimal(1),
        currency='USDT', exchange='BINANCE'))
    prices = MarketReferencePriceStore()
    driver = None
    if args.enable_orders:
        client, driver = build_readonly_client(args.symbol, allow_demo_orders=True)
        client.risk_manager = PreTradeRiskManager(
            client.client_id, client.position_manager, prices,
            instrument_limits={instrument_id: RiskLimits(
                max_order_quantity=Decimal(args.max_order_quantity),
                max_abs_position=Decimal(args.max_position),
                max_order_notional=Decimal(args.max_order_notional),
                max_abs_position_notional=Decimal(args.max_position_notional),
                max_market_age_ns=180 * 1_000_000_000,
            )},
        )
    else:
        client = RecordingExecutionClient('recording-only')
    strategy = EmaCrossTargetStrategy('ema-binance-live', EmaCrossConfig(
        fast_period=args.fast, slow_period=args.slow,
        long_quantity=Decimal(args.quantity),
        short_quantity=-Decimal(args.quantity), skip_single_price=False))
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE,
                                   position_manager=(client.position_manager
                                                     if args.enable_orders else None))
    runner.add_market_observer(prices)
    runner.add_data_feed('binance-kline', feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(DataBinding('primary_bar', 'binance-kline', instrument_id,
                                   DataType.BAR, args.interval.upper()),),
        execution_routes=(ExecutionRoute('position', client.client_id, instrument_id),),
    )
    return runner, strategy, client, driver, instrument_id


def main():
    parser = argparse.ArgumentParser(description='Binance U本位期货EMA')
    parser.add_argument('--environment', choices=('demo', 'live'), default='demo')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', choices=('1-MINUTE', '3-MINUTE', '5-MINUTE',
                        '15-MINUTE', '30-MINUTE', '1-HOUR'), default='1-MINUTE')
    parser.add_argument('--fast', type=int, default=3)
    parser.add_argument('--slow', type=int, default=5)
    parser.add_argument('--quantity', default='0.001')
    parser.add_argument('--price-precision', type=int, default=2)
    parser.add_argument('--size-precision', type=int, default=6)
    parser.add_argument('--price-increment', default='0.01')
    parser.add_argument('--ws-base-url')
    parser.add_argument('--timeout', type=float, default=0.0)
    parser.add_argument('--enable-orders', action='store_true')
    parser.add_argument('--confirm-demo', action='store_true')
    parser.add_argument('--max-order-quantity', default='0.001')
    parser.add_argument('--max-position', default='0.002')
    parser.add_argument('--max-order-notional', default='200')
    parser.add_argument('--max-position-notional', default='400')
    args = parser.parse_args()
    if args.fast <= 0 or args.slow <= args.fast or Decimal(args.quantity) <= 0:
        parser.error('EMA周期或目标数量无效')
    runner, strategy, client, driver, instrument_id = build_runner(args)
    if args.enable_orders:
        client.start()
        try:
            # 首次运行只接受完整空活动订单与空仓；恢复旧订单另走人工对账。
            orders = driver.reconcile_active_orders()
            if orders.orders:
                raise RuntimeError('DEMO账户存在活动订单，拒绝自动启动策略')
            positions = client.position_manager.snapshot().account_positions
            if any(quantity != 0 for quantity in positions.values()):
                raise RuntimeError('DEMO账户存在持仓，拒绝自动启动策略')
            client.arm_demo(client.DEMO_CONFIRMATION)
        except BaseException:
            client.stop()
            raise
    print(f'Binance EMA: {instrument_id} mode={"DEMO订单" if args.enable_orders else "Recording"}')
    started = time.monotonic()
    reported = 0
    requests_seen = 0
    next_heartbeat = time.monotonic() + 10
    try:
        runner.start()
        while args.timeout <= 0 or time.monotonic() - started < args.timeout:
            if args.enable_orders:
                reports = client.backend.reports
                for report in reports[reported:]:
                    print(f'执行回报: {report}')
                reported = len(reports)
                if not driver.node.is_running():
                    client.mark_disconnected('trading_node_stopped')
                    raise RuntimeError('DEMO交易节点已停止')
                if time.monotonic() >= next_heartbeat:
                    client.refresh_account_state()
                    next_heartbeat = time.monotonic() + 10
            else:
                requests = client.requests
                for request in requests[requests_seen:]:
                    print(f'Recording目标请求: {request}')
                requests_seen = len(requests)
            time.sleep(1 if args.enable_orders else 0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if args.enable_orders:
            try:
                client.set_risk_mode('HALTED', cancel_active_orders=True)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if not driver.reconcile_active_orders().orders:
                        break
                    time.sleep(1)
                else:
                    print('停机时柜台仍有活动订单，必须人工核对')
            except Exception as error:
                client.disarm('shutdown_query_failed')
                print(f'停机撤单或查询未完成，必须人工核对: {error}')
        runner.stop()
    print(f'已停止: bars_seen={strategy.bars_seen} bars_used={strategy.bars_used}')


if __name__ == '__main__':
    main()
