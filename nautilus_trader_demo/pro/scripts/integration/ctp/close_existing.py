"""独立SimNow一手平仓工具：默认只读检查，显式授权才发单。"""

from contextlib import ExitStack
from datetime import datetime
from decimal import Decimal, InvalidOperation
import argparse
import json
import math
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.trader.execution.contracts import (
    ExecutionReportType, OrderIntent, OrderSide, OrderType, PositionEffect,
)
from bomber.framework.trader.execution.ctp import CtpNativeTraderDriver, CtpTdApiTransport
from bomber.framework.trader.runtime.ctp import account_lock, occupied_positions

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLIENT_ID = "ctp-close-existing-one"


def decimal_arg(value):
    try:
        return Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("请输入有效数字") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--exchange", choices=("SHFE", "INE"), required=True)
    parser.add_argument("--side", choices=("BUY", "SELL"), help="提交时必填；BUY平空仓，SELL平多仓")
    parser.add_argument("--expected-trading-day", help="提交时必填：本次TD实际交易日YYYYMMDD")
    parser.add_argument("--price", type=decimal_arg, help="提交时必填：本次核对的限价")
    parser.add_argument("--price-increment", type=decimal_arg, required=True)
    parser.add_argument("--multiplier", type=decimal_arg, required=True)
    parser.add_argument("--max-notional", type=decimal_arg, default=Decimal(50000))
    parser.add_argument("--query-timeout", type=float, default=30)
    parser.add_argument("--wait-seconds", type=float, default=8)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    args = parser.parse_args(argv)
    if not args.connect:
        parser.error("须显式--connect；默认只读检查")
    if args.submit != args.confirm_simnow:
        parser.error("发单须同时指定--submit --confirm-simnow")
    if args.submit and (args.side is None or args.expected_trading_day is None or args.price is None):
        parser.error("提交须明确--side、--expected-trading-day和--price")
    if args.expected_trading_day is not None:
        try:
            day = datetime.strptime(args.expected_trading_day, "%Y%m%d")
            if day.strftime("%Y%m%d") != args.expected_trading_day:
                raise ValueError("日期格式错误")
        except ValueError:
            parser.error("TD交易日须为有效YYYYMMDD")
    if (any(not value.is_finite() or value <= 0 for value in
            (args.price_increment, args.multiplier, args.max_notional))
            or any(not math.isfinite(value) or value <= 0 for value in
                (args.query_timeout, args.wait_seconds))):
        parser.error("步长、乘数、名义金额、超时须为正且有限")
    if args.price is not None and (not args.price.is_finite() or args.price <= 0
            or args.price % args.price_increment != 0
            or args.price * args.multiplier > args.max_notional):
        parser.error("限价须为正、符合步长且在名义金额上限内")
    return args


def inspect_account(driver, instrument, args):
    """分别核对权威总仓和今昨明细，不依赖自然日／旧策略账本。"""
    if args.expected_trading_day and driver.trading_day != args.expected_trading_day:
        raise RuntimeError(f"TD交易日变化: actual={driver.trading_day} expected={args.expected_trading_day}")
    account = driver.reconcile_account_state()
    if driver.reconcile_active_orders().orders:
        raise RuntimeError("全账户存在活动订单，拒绝平仓")
    balance = account.balances.get("CNY")
    if balance is None or balance.available <= 0:
        raise RuntimeError("未取得可用CNY权威资金")
    gross = occupied_positions(driver.transport)
    print(f"权威总仓: TD交易日={driver.trading_day} 多仓/空仓={gross}", flush=True)
    actual = gross.get(str(instrument), (Decimal(0), Decimal(0)))
    if set(gross) != {str(instrument)} or actual not in {(Decimal(1), Decimal(0)), (Decimal(0), Decimal(1))}:
        raise RuntimeError("仅允许全账户这一合约恰好1手单向仓；无仓、其他仓位或双向仓均拒绝")
    side = OrderSide.SELL if actual[0] else OrderSide.BUY
    if args.side and side.value != args.side:
        raise RuntimeError(f"持仓方向与指定平仓方向不一致，当前需要{side.value}")
    rows = driver.transport.query_position_details()
    totals = {}
    selected = []
    for row in rows:
        key = f"{row['InstrumentID']}.{row['ExchangeID']}"
        long_qty, short_qty = totals.get(key, (Decimal(0), Decimal(0)))
        quantity = row["Position"]
        totals[key] = ((long_qty + quantity, short_qty) if row["PosiDirection"] == "2"
            else (long_qty, short_qty + quantity))
        if quantity and key == str(instrument):
            selected.append(row)
    if {key: value for key, value in totals.items() if any(value)} != gross:
        raise RuntimeError("查询期间总仓与明细不一致，拒绝平仓")
    if len(selected) != 1:
        raise RuntimeError("一手仓位无法唯一确定今昨属性，拒绝平仓")
    detail = selected[0]
    print(f"权威明细: {dict(detail)}", flush=True)
    # 编码来自本项目CTP头文件：PSD_Today='1'，PSD_History='2'。
    effect = {"1": PositionEffect.CLOSE_TODAY, "2": PositionEffect.CLOSE_YESTERDAY}.get(
        str(detail.get("PositionDate", "")))
    if effect is None:
        raise RuntimeError("缺少有效PositionDate，不能推断平今／平昨；请核对并重编本次bomber_ctp_td绑定")
    if str(detail.get("HedgeFlag", "")) != "1":
        raise RuntimeError("该工具仅支持投机仓，不处理套保／套利仓")
    return driver.trading_day, side, effect, gross


def wait_no_active(driver, timeout):
    deadline = time.monotonic() + timeout
    while driver.reconcile_active_orders().orders:
        if time.monotonic() >= deadline:
            raise RuntimeError("平仓委托仍活动，需要核对撤单")
        time.sleep(0.2)


def run_session(args, transport):
    if transport.broker_id != "9999":
        raise ValueError("仅允许SimNow BrokerID=9999")
    instrument = InstrumentId.from_str(f"{args.symbol}.{args.exchange}")
    reports, lost = [], []
    driver = CtpNativeTraderDriver(CLIENT_ID, transport.account_id, transport,
        enable_simnow_orders=args.submit, max_session_orders=1, disconnect_handler=lost.append)
    attempted = False
    result = {"status": "failed", "orders_submitted": 0, "final_gross": None,
        "final_active_orders": None, "cleanup_errors": []}
    with ExitStack() as resources:
        resources.enter_context(account_lock(transport.account_id))
        resources.enter_context(account_lock(transport.account_id, namespace="bomber-main-ema"))
        resources.callback(driver.stop)
        try:
            driver.start(reports.append)
            first = inspect_account(driver, instrument, args)
            day, side, effect, gross = first
            result.update(trading_day=day, instrument=str(instrument), side=side.value,
                position_effect=effect.value)
            print(f"平仓计划: {instrument} {side.value} {effect.value} 1手 限价={args.price}", flush=True)
            if not args.submit:
                print("仅预检，报单0；显式--submit --confirm-simnow才允许模拟平仓", flush=True)
                result["status"] = "prechecked"
            else:
                if inspect_account(driver, instrument, args) != first or lost:
                    raise RuntimeError("报单前会话或仓位变化，拒绝提交")
                intent = OrderIntent(strategy_id=CLIENT_ID, backend_id=CLIENT_ID,
                    instrument_id=instrument, side=side, quantity=Decimal(1), order_type=OrderType.LIMIT,
                    price=args.price, position_effect=effect, reduce_only=True)
                attempted = True
                driver.submit_order(intent)
                deadline = time.monotonic() + args.wait_seconds
                while not any(report.report_type in {ExecutionReportType.FILLED, ExecutionReportType.REJECTED}
                        for report in reports):
                    if lost or time.monotonic() >= deadline:
                        break
                    time.sleep(0.2)
                if lost:
                    raise RuntimeError("平仓期间TD断线，需要重新只读核对")
                driver.cancel_strategy(CLIENT_ID)
                wait_no_active(driver, args.query_timeout)
                for report in reports:
                    print(f"CTP回报: {report}", flush=True)
                if not any(report.report_type is ExecutionReportType.FILLED
                        and report.position_effect is effect and report.filled_quantity == 1
                        for report in reports):
                    raise RuntimeError("未取得完整平仓成交，不自动重发，请核对本次回报与总仓")
                deadline = time.monotonic() + args.query_timeout
                while occupied_positions(transport):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("成交后权威总仓尚未归零")
                    time.sleep(0.2)
                result["status"] = "passed"
        except BaseException as error:
            result["failure"] = str(error)
            raise
        finally:
            errors = result["cleanup_errors"]
            if attempted and not lost:
                try:
                    driver.cancel_strategy(CLIENT_ID)
                    wait_no_active(driver, args.query_timeout)
                except Exception as error:
                    errors.append(str(error))
            try:
                if driver.trading_day is not None:
                    result["final_active_orders"] = len(driver.reconcile_active_orders().orders)
                    result["final_gross"] = occupied_positions(transport)
                    driver.reconcile_account_state()
                if result["final_active_orders"] or (args.submit and result["status"] == "passed"
                        and result["final_gross"]):
                    raise RuntimeError("最终账户查询不满足无活动订单／平仓归零条件")
            except Exception as error:
                errors.append(str(error))
            result["orders_submitted"] = driver.submitted_orders
            if errors:
                result["status"] = "failed"
            print("平仓工具结果: " + json.dumps(result, ensure_ascii=False, default=str), flush=True)
            if errors:
                raise RuntimeError("平仓停机核对失败: " + "; ".join(errors))
    return result


def main(argv=None):
    args = parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")

    def required(name):
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(f"缺少环境变量: {name}")
        return value

    investor = required("CTP_ACCOUNT_ID")
    transport = CtpTdApiTransport(client_id=CLIENT_ID, account_id=investor,
        front=required("CTP_TD_ADDRESS"), broker_id=required("CTP_BROKER_ID"), investor_id=investor,
        password=required("CTP_PASSWORD"), app_id=os.getenv("CTP_APP_ID", ""),
        auth_code=os.getenv("CTP_AUTH_CODE", ""),
        flow_path=os.getenv("CTP_CLOSE_FLOW_PATH", "/tmp/bomber-ctp-close-existing"),
        production_mode=os.getenv("CTP_PRODUCTION_MODE", "true").lower() in {"1", "true", "yes", "on"},
        timeout_seconds=args.query_timeout)
    print(f"SimNow平仓工具: TD={transport.front} user=***{investor[-2:]} submit={args.submit}", flush=True)
    return run_session(args, transport)


if __name__ == "__main__":
    main()
