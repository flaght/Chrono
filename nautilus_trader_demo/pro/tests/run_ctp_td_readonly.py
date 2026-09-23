"""CTP只读柜台装配的无网络烟测。"""

from __future__ import annotations

import os
from unittest.mock import patch

from examples.single_ema.ctp_td_readonly import build_readonly_driver
from run_p4_ctp_positions import PositionTdApi
from run_p4_ctp_driver import _intent


def main() -> None:
    environment = {
        "CTP_TD_ADDRESS": "tcp://fake:1234", "CTP_BROKER_ID": "9999",
        "CTP_ACCOUNT_ID": "demo", "CTP_PASSWORD": "fake-password",
        "CTP_APP_ID": "fake-app", "CTP_AUTH_CODE": "fake-auth",
    }
    with patch.dict(os.environ, environment):
        driver = build_readonly_driver(timeout=0.05, td_api_base=PositionTdApi)
        driver.start(lambda report: None)
        try:
            assert driver.reconcile_position_detail().positions["rb2704.SHFE"].net_quantity == 1
            assert driver.reconcile_account_state().balances["CNY"].available == 80000
            assert len(driver.reconcile_active_orders().orders) == 1
            try:
                driver.submit_order(_intent())
            except RuntimeError as error:
                assert "默认禁单" in str(error)
            else:
                raise AssertionError("只读探针不允许报单")
            assert "insert" not in driver.transport._api.request_names
        finally:
            driver.stop()
    print("CTP只读探针通过：登录/结算确认及三项权威查询完成，报单保持禁用")


if __name__ == "__main__":
    main()
