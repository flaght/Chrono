"""CTP 只读入口迁移烟测：不读取真实凭据、不连接网络、不发柜台订单。"""

from __future__ import annotations

from decimal import Decimal
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .run_p4_ctp_driver import _intent
from .run_p4_ctp_td_transport import FakeTdApi


def main() -> None:
    # 直接运行本文件时，tests 本身就是 Python 的首个搜索目录。
    # 跨目录入口按文件位置加载，不修改全局导入路径。
    entry_path = Path(__file__).resolve().parent.parent / "scripts/integration/ctp/td_readonly.py"
    spec = spec_from_file_location("ctp_td_readonly_under_test", entry_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 CTP 只读入口: {entry_path}")
    current = module_from_spec(spec)
    # 导入时屏蔽 dotenv 读取，测试始终只使用下面的假凭据。
    with patch("dotenv.load_dotenv") as load_environment:
        spec.loader.exec_module(current)
        load_environment.assert_called_once_with(Path(__file__).resolve().parent.parent / ".env")
    assert current.PROJECT_ROOT == Path(__file__).resolve().parent.parent

    with TemporaryDirectory(prefix="ctp-readonly-smoke-") as flow_directory:
        environment = {
            "CTP_TD_ADDRESS": "tcp://fake:1234", "CTP_BROKER_ID": "9999",
            "CTP_ACCOUNT_ID": "demo", "CTP_PASSWORD": "fake-password",
            "CTP_APP_ID": "fake-app", "CTP_AUTH_CODE": "fake-auth",
            "CTP_PRODUCTION_MODE": "true", "CTP_TD_FLOW_PATH": flow_directory,
        }
        with patch.dict(os.environ, environment, clear=True):
            driver = current.build_readonly_driver(timeout=0.05, td_api_base=FakeTdApi)
            driver.start(lambda report: None)
            api = driver.transport._api
            try:
                assert driver.reconcile()["rb2704.SHFE"] == Decimal(1)
                assert driver.reconcile_account_state().balances["CNY"].available == 80000
                assert len(driver.reconcile_active_orders().orders) == 1
                assert api.request_names[:4] == ["create", "auth", "login", "settlement"]
                try:
                    driver.submit_order(_intent())
                except RuntimeError as error:
                    assert "默认禁单" in str(error)
                else:
                    raise AssertionError("只读入口不允许报单")
                assert "insert" not in api.request_names
                assert "cancel" not in api.request_names
            finally:
                driver.stop()
            assert api.exited
    print("CTP只读入口通过：配置根目录、登录/结算确认、三项查询与禁单正常")


if __name__ == "__main__":
    main()
