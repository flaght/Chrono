#!/usr/bin/env python3
"""One-shot SimNow order probe with account and target-position guards."""

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


def _price_arg(value: str, option: str) -> Decimal:
    if not value.strip():
        raise argparse.ArgumentTypeError(
            f"{option} 为空；请先输入已核对的数字限价"
        )
    return decimal_arg(value)


def open_price_arg(value: str) -> Decimal:
    return _price_arg(value, "--price（开仓限价）")


def close_price_arg(value: str) -> Decimal:
    return _price_arg(value, "--close-price（平今限价）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTP SimNow 单笔限价单探针")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--exchange", choices=("SHFE", "INE"), required=True)
    parser.add_argument("--side", choices=("BUY", "SELL"), required=True)
    parser.add_argument("--price", type=open_price_arg)
    parser.add_argument("--close-price", type=close_price_arg,
                        help="往返测试的平今限价；须与 --round-trip 同时指定")
    parser.add_argument("--price-increment", type=decimal_arg, required=True)
    parser.add_argument("--multiplier", type=decimal_arg, required=True)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--max-notional", type=decimal_arg, default=Decimal("50000"))
    parser.add_argument("--cancel-after", type=float, default=3.0)
    parser.add_argument("--query-timeout", type=float, default=30.0)
    parser.add_argument("--allow-other-positions", action="store_true",
                        help="允许其他合约已有仓位；测试合约仍须空仓且全账户无活动订单")
    parser.add_argument("--round-trip", action="store_true",
                        help="开仓成交且柜台仓位为1手后，尝试反向平今并确认回到空仓")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    args = parser.parse_args()
    if args.submit != args.confirm_simnow:
        parser.error("发单须同时指定 --submit 和 --confirm-simnow")
    if args.submit and args.price is None:
        parser.error("发单必须提供 --price")
    if args.round_trip and (not args.submit or args.close_price is None):
        parser.error("往返测试须指定 --round-trip --close-price --submit --confirm-simnow")
    if args.close_price is not None and not args.round_trip:
        parser.error("--close-price 仅与 --round-trip 一起使用")
    if (
        not args.price_increment.is_finite() or args.price_increment <= 0
        or not args.multiplier.is_finite() or args.multiplier <= 0
        or args.quantity != 1
        or not args.max_notional.is_finite() or args.max_notional <= 0
        or args.cancel_after <= 0 or args.query_timeout <= 0
    ):
        parser.error("只允许1手、有效价格步长和名义金额上限内的限价单")
    for price in (args.price, args.close_price):
        if price is not None and (
            not price.is_finite() or price <= 0
            or price % args.price_increment != 0
            or price * args.multiplier > args.max_notional
        ):
            parser.error("限价须为有效价格步长的整数倍且在名义金额上限内")
    return args


def assert_account_ready(
    driver: CtpNativeTraderDriver,
    transport: CtpTdApiTransport,
    instrument_id: InstrumentId,
    *,
    allow_other_positions: bool,
) -> dict[str, tuple[Decimal, Decimal]]:
    orders = driver.reconcile_active_orders()
    driver.reconcile_account_state()
    gross = transport.query_gross_positions()
    if orders.orders:
        raise RuntimeError(f"账户已有{len(orders.orders)}笔活动订单，拒绝发送测试单")
    occupied = {key: amounts for key, amounts in gross.items() if any(amounts)}
    if str(instrument_id) in occupied:
        raise RuntimeError(f"测试合约{instrument_id}已有多/空仓，拒绝发送开仓测试单")
    if occupied and not allow_other_positions:
        raise RuntimeError(
            f"账户已有其他合约仓位 {occupied}；仅测试其他合约时可显式指定 --allow-other-positions"
        )
    return occupied


def occupied_positions(transport: CtpTdApiTransport) -> dict[str, tuple[Decimal, Decimal]]:
    return {
        key: amounts for key, amounts in transport.query_gross_positions().items()
        if any(amounts)
    }


def wait_no_active_orders(driver: CtpNativeTraderDriver, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not driver.reconcile_active_orders().orders:
            return
        time.sleep(1)
    raise RuntimeError("测试委托仍在活动订单中，须在交易终端人工核对及撤单")


def report_summary(reports: list) -> None:
    print(f"订单回报数={len(reports)}")
    for report in reports:
        print(f"CTP回报: {report}")


def assert_other_positions_unchanged(
    occupied: dict[str, tuple[Decimal, Decimal]],
    instrument_id: InstrumentId,
    baseline: dict[str, tuple[Decimal, Decimal]],
) -> None:
    others = {key: value for key, value in occupied.items() if key != str(instrument_id)}
    if others != baseline:
        raise RuntimeError(f"其他合约仓位发生变化: 测试前={baseline} 测试后={others}")


def wait_target_position(
    transport: CtpTdApiTransport,
    instrument_id: InstrumentId,
    expected: tuple[Decimal, Decimal],
    other_positions: dict[str, tuple[Decimal, Decimal]],
    timeout: float,
) -> dict[str, tuple[Decimal, Decimal]]:
    deadline = time.monotonic() + timeout
    while True:
        occupied = occupied_positions(transport)
        assert_other_positions_unchanged(occupied, instrument_id, other_positions)
        if occupied.get(str(instrument_id), (Decimal(0), Decimal(0))) == expected:
            return occupied
        if time.monotonic() >= deadline:
            return occupied
        time.sleep(1)


def main() -> None:
    args = parse_args()
    instrument_id = InstrumentId.from_str(f"{args.symbol}.{args.exchange}")
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
        initial_positions = assert_account_ready(
            driver, transport, instrument_id,
            allow_other_positions=args.allow_other_positions,
        )
        if initial_positions:
            print(f"已有其他合约仓位（测试期间须保持不变）: {initial_positions}")
        if args.price is None:
            print("测试合约空仓及全账户无活动订单确认；预检完成，未发送订单。发单时须提供 --price")
            return
        intent = OrderIntent(
            strategy_id=STRATEGY_ID,
            backend_id=DRIVER_ID,
            instrument_id=instrument_id,
            side=OrderSide(args.side),
            quantity=Decimal(1),
            order_type=OrderType.LIMIT,
            price=args.price,
            position_effect=PositionEffect.OPEN,
        )
        print(
            f"测试合约空仓及全账户无活动订单确认：{intent.instrument_id} "
            f"{intent.side.value} OPEN 1手 限价={args.price} "
            f"名义金额={args.price * args.multiplier}"
        )
        if not args.submit:
            print("预检完成；未发送订单。发单需 --submit --confirm-simnow")
            return
        # Recheck immediately before submission; this probe requires exclusive
        # use of the SimNow account during the test.
        positions_before = assert_account_ready(
            driver, transport, instrument_id,
            allow_other_positions=args.allow_other_positions,
        )
        if positions_before != initial_positions:
            raise RuntimeError("预检后其他合约仓位发生变化，拒绝发送测试单")
        submitted = True
        # A send call can fail after the counter receives the order. Keep the
        # cleanup path active even when submission raises ambiguously.
        driver.submit_order(intent)
        print("开仓报单请求已发送，等待柜台回报")
        time.sleep(args.cancel_after)
        if disconnected:
            raise RuntimeError(f"交易连接已断开: {disconnected[-1]}")
        driver.cancel_strategy(STRATEGY_ID)
        wait_no_active_orders(driver, args.query_timeout)
        report_summary(reports)
        positions_after = occupied_positions(transport)
        assert_other_positions_unchanged(positions_after, instrument_id, positions_before)
        rejected = [report for report in reports if report.report_type is ExecutionReportType.REJECTED]
        if rejected:
            raise RuntimeError(f"CTP报单被拒绝: {rejected[-1].reason}")
        if not reports:
            raise RuntimeError("未收到CTP订单回报；须在交易终端核对委托状态")
        if args.round_trip:
            expected = (
                (Decimal(1), Decimal(0)) if intent.side is OrderSide.BUY
                else (Decimal(0), Decimal(1))
            )
            open_fill_seen = any(
                report.report_type is ExecutionReportType.FILLED
                and report.position_effect is PositionEffect.OPEN
                and report.filled_quantity == 1
                for report in reports
            )
            if not open_fill_seen:
                actual = positions_after.get(str(instrument_id), (Decimal(0), Decimal(0)))
                print(f"最终净仓: {driver.reconcile()}")
                if actual == (Decimal(0), Decimal(0)):
                    raise RuntimeError("买开单已结束且未成交；未发送平今单，请按当前盘口重新选择限价")
                raise RuntimeError(f"柜台已有测试合约仓位但缺开仓成交回报，须人工核对: {actual}")
            positions_after = wait_target_position(
                transport, instrument_id, expected, positions_before, args.query_timeout,
            )
            actual = positions_after.get(str(instrument_id), (Decimal(0), Decimal(0)))
            if actual != expected:
                raise RuntimeError(
                    f"开仓未确认恰好成交1手，停止往返测试: {instrument_id} 多/空={actual}"
                )
            close_intent = OrderIntent(
                strategy_id=STRATEGY_ID,
                backend_id=DRIVER_ID,
                instrument_id=instrument_id,
                side=OrderSide.SELL if intent.side is OrderSide.BUY else OrderSide.BUY,
                quantity=Decimal(1),
                order_type=OrderType.LIMIT,
                price=args.close_price,
                position_effect=PositionEffect.CLOSE_TODAY,
                reduce_only=True,
            )
            print(f"开仓成交及柜台仓位确认；发送平今单: {close_intent.side.value} 1手 限价={args.close_price}")
            driver.submit_order(close_intent)
            time.sleep(args.cancel_after)
            if disconnected:
                raise RuntimeError(f"平仓期间交易连接断开: {disconnected[-1]}")
            driver.cancel_strategy(STRATEGY_ID)
            wait_no_active_orders(driver, args.query_timeout)
            report_summary(reports)
            close_fill_seen = any(
                report.report_type is ExecutionReportType.FILLED
                and report.position_effect is PositionEffect.CLOSE_TODAY
                and report.filled_quantity == 1
                for report in reports
            )
            if not close_fill_seen:
                current_positions = occupied_positions(transport)
                assert_other_positions_unchanged(current_positions, instrument_id, positions_before)
                print(f"最终净仓: {driver.reconcile()}")
                raise RuntimeError(
                    "平今单没有成交回报；须人工核对测试合约仓位: "
                    f"{current_positions.get(str(instrument_id), (Decimal(0), Decimal(0)))}"
                )
            final_positions = wait_target_position(
                transport, instrument_id, (Decimal(0), Decimal(0)),
                positions_before, args.query_timeout,
            )
            if final_positions.get(str(instrument_id), (Decimal(0), Decimal(0))) != (0, 0):
                raise RuntimeError(
                    f"平今未完成，测试合约仍有仓位，须人工核对: {final_positions.get(str(instrument_id))}"
                )
            print("往返测试通过：开仓与平今均成交，测试合约空仓，其他合约仓位未变")
        print(f"最终净仓: {driver.reconcile()}")
        if not args.round_trip:
            print("单笔测试结束；若开仓成交，测试合约仓位不会自动平仓")
    finally:
        if submitted and not disconnected:
            try:
                driver.cancel_strategy(STRATEGY_ID)
            except Exception as error:
                print(f"退出时撤单未确认，须人工核对: {error}")
        driver.stop()


if __name__ == "__main__":
    main()
