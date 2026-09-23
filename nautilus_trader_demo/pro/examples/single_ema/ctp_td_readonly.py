#!/usr/bin/env python3
"""CTP TraderApi只读联调：登录/结算确认后查询仓位、资金和活动订单。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from strategy.execution.ctp.native_driver import CtpNativeTraderDriver
from strategy.execution.ctp.td_transport import CtpTdApiTransport


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"缺少环境变量: {name}")
    return value


def build_readonly_driver(
    *, timeout: float = 15.0, td_api_base: type | None = None,
) -> CtpNativeTraderDriver:
    """没有测试报单开关；即使误调用submit_order也必须被Driver拒绝。"""
    investor = _required("CTP_ACCOUNT_ID")
    transport = CtpTdApiTransport(
        client_id="ctp-td-readonly",
        account_id=investor,
        front=_required("CTP_TD_ADDRESS"),
        broker_id=_required("CTP_BROKER_ID"),
        investor_id=investor,
        password=_required("CTP_PASSWORD"),
        app_id=os.getenv("CTP_APP_ID", ""),
        auth_code=os.getenv("CTP_AUTH_CODE", ""),
        flow_path=os.getenv("CTP_TD_FLOW_PATH", "/tmp/bomber-ctp-td-readonly"),
        timeout_seconds=timeout,
        td_api_base=td_api_base,
    )
    return CtpNativeTraderDriver(
        "ctp-td-readonly", investor, transport,
        enable_test_orders=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CTP原生TraderApi只读柜台探针")
    parser.add_argument("--connect", action="store_true", help="显式允许连接柜台")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    if not args.connect:
        parser.error("必须显式传入--connect；本探针不会发单")
    if args.timeout <= 0:
        parser.error("--timeout必须大于零")
    driver = build_readonly_driver(timeout=args.timeout)
    print("连接CTP TraderApi只读探针：将执行登录、结算确认与三项权威查询；不发单")
    driver.start(lambda report: None)
    try:
        position_event = driver.reconcile_position_detail()
        account = driver.reconcile_account_state()
        orders = driver.reconcile_active_orders()
        print(
            f"CTP只读结果: position_revision={position_event.revision} "
            f"instruments={len(position_event.positions)} "
            f"account_revision={account.revision} currencies={tuple(account.balances)} "
            f"orders_revision={orders.revision} active_orders={len(orders.orders)}"
        )
        for instrument, position in position_event.positions.items():
            print(
                f"仓位 {instrument}: long_today={position.long_today} "
                f"long_yesterday={position.long_yesterday} "
                f"short_today={position.short_today} "
                f"short_yesterday={position.short_yesterday}"
            )
    finally:
        driver.stop()


if __name__ == "__main__":
    main()
