#!/usr/bin/env python3
"""SimNow EMA受控报单：只接受官网仿真前置、初始空仓及空活动订单。"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

from dotenv import load_dotenv
from market.basic.base import DataType, InstrumentId, InstrumentMeta
from market.stream import TradeTickBarFeed
from market.stream.ctp import CtpLiveDataFeed, CtpMdConfig
from trader import (ControlledLiveExecutionClient, DataBinding, ExecutionRoute,
                      MarketReferencePriceStore, NautilusLiveExecutionBackend,
                      PositionManager, PreTradeRiskManager, RiskLimits, RuntimeMode,
                      UnifiedStrategyRunner)
from trader.execution.contracts import OrderSide, OrderType, PositionEffect
from trader.execution.ctp import (CtpClosePlanner, CtpExecutionAccounting,
                                    CtpNativeTraderDriver, CtpPositionLedger,
                                    CtpTdApiTransport)
from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy

load_dotenv(Path(__file__).resolve().parents[2] / '.env')


def required(name):
    value = os.getenv(name)
    if not value:
        raise SystemExit(f'缺少环境变量: {name}')
    return value


class CtpLimitPlanner:
    """CTP开平今昨仓规划后，为每笔订单附上有保护的限价。"""
    def __init__(self, ledger, positions, prices, tick_size, offset_ticks):
        self._planner = CtpClosePlanner(ledger, close_today_first=True)
        self._positions = positions
        self._prices = prices
        self._tick_size = tick_size
        self._offset_ticks = offset_ticks

    def plan(self, request):
        result = []
        planned = tuple(self._planner.plan(request))
        closing_instruments = {
            order.instrument_id for order in planned
            if order.position_effect is not PositionEffect.OPEN
        }
        for order in planned:
            if self._positions.working_quantity(request.client_id, order.instrument_id):
                continue
            # A reversal closes the old side first. The next bar repeats the
            # target after the close fill updates the position ledger.
            if order.position_effect is PositionEffect.OPEN and order.instrument_id in closing_instruments:
                continue
            reference = self._prices.get(order.instrument_id)
            if reference is None:
                raise RuntimeError('CTP缺少行情参考价，拒绝报单')
            raw = reference.price + (self._tick_size * self._offset_ticks
                                     * (1 if order.side is OrderSide.BUY else -1))
            rounding = ROUND_CEILING if order.side is OrderSide.BUY else ROUND_FLOOR
            price = (raw / self._tick_size).to_integral_value(rounding=rounding) * self._tick_size
            if price <= 0:
                raise RuntimeError('CTP限价无效')
            result.append(replace(order, order_type=OrderType.LIMIT, price=price))
        return tuple(result)


def build_runner(args):
    if not args.enable_orders or not args.confirm_simnow:
        raise SystemExit('SimNow订单入口须同时指定 --enable-orders --confirm-simnow；只看行情请用 ema_ctp_live.py')
    if required('CTP_BROKER_ID') != '9999':
        raise SystemExit('SimNow报单仅允许BrokerID=9999')
    investor = required('CTP_ACCOUNT_ID')
    symbol = args.symbol or required('CTP_SYMBOL')
    if args.exchange.upper() not in {'SHFE', 'INE'}:
        raise SystemExit('首版SimNow EMA订单只支持SHFE/INE明确平今平昨规则')
    instrument = InstrumentId.from_str(f'{symbol}.{args.exchange.upper()}')
    multiplier = Decimal(args.multiplier)
    tick_size = Decimal(args.price_increment)
    if (args.fast <= 0 or args.slow <= args.fast or Decimal(args.quantity) <= 0
            or Decimal(args.quantity) != Decimal(args.quantity).to_integral_value()
            or multiplier <= 0 or tick_size <= 0 or args.limit_offset_ticks < 0):
        raise SystemExit('EMA周期、手数、合约乘数、价格步长或限价偏移无效')
    front = required('CTP_TD_ADDRESS')
    transport = CtpTdApiTransport(
        client_id='ctp-simnow', account_id=investor,
        front=front, broker_id='9999', investor_id=investor,
        password=required('CTP_PASSWORD'),
        app_id=os.getenv('CTP_APP_ID', ''), auth_code=os.getenv('CTP_AUTH_CODE', ''),
        flow_path=os.getenv('CTP_TD_FLOW_PATH', '/tmp/bomber-ctp-ema-td'),
        production_mode=os.getenv('CTP_PRODUCTION_MODE', 'true').lower()
        in {'1', 'true', 'yes', 'on'},
        timeout_seconds=args.query_timeout,
    )
    holder = {}
    driver = CtpNativeTraderDriver(
        'ctp-simnow', investor, transport, enable_simnow_orders=True,
        disconnect_handler=lambda reason: holder['client'].mark_disconnected(reason),
    )
    backend = NautilusLiveExecutionBackend('ctp-simnow', driver)
    ledger = CtpPositionLedger()
    positions = PositionManager()
    prices = MarketReferencePriceStore()
    planner = CtpLimitPlanner(ledger, positions, prices, tick_size, args.limit_offset_ticks)
    risk = PreTradeRiskManager(
        'ctp-simnow', positions, prices,
        instrument_limits={instrument: RiskLimits(
            max_order_quantity=Decimal(args.max_order_quantity),
            max_abs_position=Decimal(args.max_position),
            max_order_notional=Decimal(args.max_order_notional),
            max_abs_position_notional=Decimal(args.max_position_notional),
            max_market_age_ns=120 * 1_000_000_000,
            contract_multiplier=multiplier,
        )},
    )
    client = ControlledLiveExecutionClient(
        'ctp-simnow', planner, backend, positions, risk, account_id=investor,
        demo_environment_check=lambda: driver.is_simnow_session,
        max_request_wall_age_ns=120 * 1_000_000_000,
    )
    holder['client'] = client
    accounting = CtpExecutionAccounting(ledger, {instrument: multiplier})
    backend.register_report_handler(accounting.on_report)
    upstream = CtpLiveDataFeed(CtpMdConfig(
        front=required('CTP_MD_ADDRESS'), broker_id='9999', user_id=investor,
        password=required('CTP_PASSWORD'),
        flow_path=os.getenv('CTP_MD_FLOW_PATH', '/tmp/bomber-ctp-ema-md'),
        production_mode=os.getenv('CTP_PRODUCTION_MODE', 'true').lower()
        in {'1', 'true', 'yes', 'on'}))
    feed = TradeTickBarFeed('CTP_EMA_1M', upstream)
    feed.register_instrument(InstrumentMeta(
        instrument_id=instrument, price_precision=args.price_precision,
        size_precision=0, price_increment=tick_size, multiplier=multiplier,
        currency='CNY', exchange=args.exchange.upper()))
    strategy = EmaCrossTargetStrategy('ema-ctp-simnow', EmaCrossConfig(
        fast_period=args.fast, slow_period=args.slow,
        long_quantity=Decimal(args.quantity), short_quantity=-Decimal(args.quantity),
        skip_single_price=False, repeat_target_each_bar=True))
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed('ctp-bars', feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(DataBinding('primary_bar', 'ctp-bars', instrument,
                                   DataType.BAR, '1-MINUTE'),),
        execution_routes=(ExecutionRoute('position', client.client_id, instrument),),
    )
    return runner, strategy, client, driver, transport, ledger


def main():
    parser = argparse.ArgumentParser(description='SimNow受控EMA交易')
    parser.add_argument('--symbol')
    parser.add_argument('--exchange', default=os.getenv('CTP_EXCHANGE', 'SHFE'))
    parser.add_argument('--fast', type=int, default=2)
    parser.add_argument('--slow', type=int, default=3)
    parser.add_argument('--quantity', default='1')
    parser.add_argument('--price-precision', type=int, default=0)
    parser.add_argument('--price-increment', default='1')
    parser.add_argument('--multiplier', default='10')
    parser.add_argument('--limit-offset-ticks', type=int, default=1)
    parser.add_argument('--max-order-quantity', default='1')
    parser.add_argument('--max-position', default='1')
    parser.add_argument('--max-order-notional', default='50000')
    parser.add_argument('--max-position-notional', default='50000')
    parser.add_argument('--query-timeout', type=float, default=15)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--enable-orders', action='store_true')
    parser.add_argument('--confirm-simnow', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0 or args.query_timeout <= 0:
        parser.error('超时必须大于零')
    runner, strategy, client, driver, transport, ledger = build_runner(args)
    client.start()
    try:
        orders = driver.reconcile_active_orders()
        if orders.orders:
            raise RuntimeError('SimNow账户已有活动订单，拒绝自动接管')
        gross = transport.query_gross_positions()
        if any(long_qty or short_qty for long_qty, short_qty in gross.values()):
            raise RuntimeError('SimNow账户已有多仓或空仓，缺成本账本，拒绝自动接管')
        client.arm_demo(client.DEMO_CONFIRMATION)
    except BaseException:
        client.stop()
        raise
    print('SimNow EMA已完成空仓、资金与活动订单权威查询；开始受控模拟报单')
    seen = 0
    next_account_check = time.monotonic() + 10
    next_position_check = time.monotonic() + 10
    try:
        runner.start()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if not driver.is_simnow_session:
                client.mark_disconnected('ctp_session_lost')
                raise RuntimeError('SimNow交易会话已断开')
            if time.monotonic() >= next_account_check:
                client.refresh_account_state()
                next_account_check = time.monotonic() + 10
            if time.monotonic() >= next_position_check:
                gross = transport.query_gross_positions()
                ledger_positions = ledger.state().positions
                for key in set(gross) | {str(item) for item in ledger_positions}:
                    snapshot = next((value for instrument, value in ledger_positions.items()
                                     if str(instrument) == key), None)
                    expected = (Decimal(0), Decimal(0)) if snapshot is None else (
                        snapshot.long_total, snapshot.short_total)
                    if gross.get(key, (Decimal(0), Decimal(0))) != expected:
                        client.mark_disconnected('position_ledger_mismatch')
                        raise RuntimeError(f'SimNow柜台与本地今昨仓账本不一致: {key}')
                next_position_check = time.monotonic() + 10
            backend_reports = client.backend.reports[seen:]
            for report in backend_reports:
                print(f'CTP执行回报: {report}')
            seen += len(backend_reports)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.set_risk_mode('HALTED', cancel_active_orders=True)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if not driver.reconcile_active_orders().orders:
                    break
                time.sleep(1)
            else:
                print('停机时SimNow柜台仍有活动订单，必须人工核对')
        except Exception as error:
            client.disarm('shutdown_query_failed')
            print(f'停机撤单或查询未完成，必须人工核对: {error}')
        runner.stop()
    print(f'已停止: bars_seen={strategy.bars_seen}, reports={seen}')


if __name__ == '__main__':
    main()
