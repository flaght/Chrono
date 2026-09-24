#!/usr/bin/env python3
"""One-shot SimNow order probe. An existing position or order blocks submission."""

from __future__ import annotations

import argparse
import os
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import load_dotenv

from market.basic.base import InstrumentId
from trader.execution.contracts import (
    ExecutionReportType,
    OrderIntent,
    OrderSide,
    OrderType,
    PositionEffect,
)
from trader.execution.ctp import CtpNativeTraderDriver, CtpTdApiTransport


load_dotenv(Path(__file__).resolve().parents[2] / ".env")
STRATEGY_ID = "ctp-simnow-one-shot"
DRIVER_ID = "ctp-simnow-one-shot"


def required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"缺少环境变量: {name}")
    return value


def decimal_arg(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("请输入有效数字") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTP SimNow 单笔限价单探针")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--exchange", choices=("SHFE", "INE"), required=True)
    parser.add_argument("--side", choices=("BUY", "SELL"), required=True)
    parser.add_argument("--price", type=decimal_arg)
    parser.add_argument("--price-increment", type=decimal_arg, required=True)
    parser.add_argument("--multiplier", type=decimal_arg, required=True)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--max-notional", type=decimal_arg, default=Decimal("50000"))
    parser.add_argument("--cancel-after", type=float, default=3.0)
    parser.add_argument("--query-timeout", type=float, default=30.0)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    args = parser.parse_args()
    if args.submit != args.confirm_simnow:
        parser.error("发单须同时指定 --submit 和 --confirm-simnow")
    if args.submit and args.price is None:
        parser.error("发单必须提供 --price")
    if (
        not args.price_increment.is_finite() or args.price_increment <= 0
        or not args.multiplier.is_finite() or args.multiplier <= 0
        or args.quantity != 1
        or not args.max_notional.is_finite() or args.max_notional <= 0
        or args.cancel_after <= 0 or args.query_timeout <= 0
    ):
        parser.error("只允许1手、有效价格步长和名义金额上限内的限价单")
    if args.price is not None and (
        not args.price.is_finite() or args.price <= 0
        or args.price % args.price_increment != 0
        or args.price * args.multiplier > args.max_notional
    ):
        parser.error("限价须为有效价格步长的整数倍且在名义金额上限内")
    return args


def assert_empty(driver: CtpNativeTraderDriver, transport: CtpTdApiTransport) -> None:
    orders = driver.reconcile_active_orders()
    driver.reconcile_account_state()
    gross = transport.query_gross_positions()
    if orders.orders:
        raise RuntimeError(f"账户已有{len(orders.orders)}笔活动订单，拒绝发送测试单")
    occupied = {key: amounts for key, amounts in gross.items() if any(amounts)}
    if occupied:
        raise RuntimeError(f"账户已有多/空仓 {occupied}，拒绝发送测试单")


def main() -> None:
    args = parse_args()
    if required("CTP_BROKER_ID") != "9999":
        raise SystemExit("仅允许SimNow BrokerID=9999")
    investor = required("CTP_ACCOUNT_ID")
    transport = CtpTdApiTransport(
        client_id=DRIVER_ID,
        account_id=investor,
        front=required("CTP_TD_ADDRESS"),
        broker_id="9999",
        investor_id=investor,
        password=required("CTP_PASSWORD"),
        app_id=os.getenv("CTP_APP_ID", ""),
        auth_code=os.getenv("CTP_AUTH_CODE", ""),
        flow_path=os.getenv("CTP_TD_FLOW_PATH", "/tmp/bomber-ctp-one-shot"),
        production_mode=os.getenv("CTP_PRODUCTION_MODE", "true").lower()
        in {"1", "true", "yes", "on"},
        timeout_seconds=args.query_timeout,
    )
    disconnected: list[str] = []
    reports = []
    driver = CtpNativeTraderDriver(
        DRIVER_ID, investor, transport,
        enable_simnow_orders=args.submit,
        disconnect_handler=disconnected.append,
    )
    submitted = False
    driver.start(reports.append)
    try:
        assert_empty(driver, transport)
        if args.price is None:
            print("空仓及空活动订单确认；预检完成，未发送订单。发单时须提供 --price")
            return
        intent = OrderIntent(
            strategy_id=STRATEGY_ID,
            backend_id=DRIVER_ID,
            instrument_id=InstrumentId.from_str(f"{args.symbol}.{args.exchange}"),
            side=OrderSide(args.side),
            quantity=Decimal(1),
            order_type=OrderType.LIMIT,
            price=args.price,
            position_effect=PositionEffect.OPEN,
        )
        print(
            f"空仓及空活动订单确认：{intent.instrument_id} "
            f"{intent.side.value} OPEN 1手 限价={args.price} "
            f"名义金额={args.price * args.multiplier}"
        )
        if not args.submit:
            print("预检完成；未发送订单。发单需 --submit --confirm-simnow")
            return
        # Recheck immediately before submission; this probe requires exclusive
        # use of the SimNow account during the test.
        assert_empty(driver, transport)
        driver.submit_order(intent)
        submitted = True
        print("报单请求已发送，等待柜台回报；若成交将留下仓位，不自动反向平仓")
        time.sleep(args.cancel_after)
        if disconnected:
            raise RuntimeError(f"交易连接已断开: {disconnected[-1]}")
        driver.cancel_strategy(STRATEGY_ID)
        deadline = time.monotonic() + args.query_timeout
        while time.monotonic() < deadline:
            if not driver.reconcile_active_orders().orders:
                break
            time.sleep(1)
        else:
            raise RuntimeError("测试委托仍在活动订单中，须在交易终端人工核对及撤单")
        print(f"订单回报数={len(reports)}")
        for report in reports:
            print(f"CTP回报: {report}")
        print(f"最终净仓: {driver.reconcile()}")
        rejected = [report for report in reports if report.report_type is ExecutionReportType.REJECTED]
        if rejected:
            raise RuntimeError(f"CTP报单被拒绝: {rejected[-1].reason}")
        if not reports:
            raise RuntimeError("未收到CTP订单回报；须在交易终端核对委托状态")
    finally:
        if submitted and not disconnected:
            try:
                driver.cancel_strategy(STRATEGY_ID)
            except Exception as error:
                print(f"退出时撤单未确认，须人工核对: {error}")
        driver.stop()


if __name__ == "__main__":
    main()
