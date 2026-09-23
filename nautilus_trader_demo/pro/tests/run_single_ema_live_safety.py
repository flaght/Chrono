"""在线EMA装配安全边界：未完成受控对账前禁止示例真实下单。"""

from examples.single_ema.ema_binance_live import _check_execution_authorization


def main() -> None:
    for environment in ("demo", "live"):
        _check_execution_authorization(environment, False, False)
        for confirm_live in (False, True):
            try:
                _check_execution_authorization(environment, True, confirm_live)
            except SystemExit as error:
                assert "暂时禁用" in str(error)
            else:
                raise AssertionError("未完成受控装配时不能启用在线EMA下单")
    print("J6安全闸门通过：BN在线EMA默认Recording，DEMO/LIVE下单均拒绝绕过受控客户端")


if __name__ == "__main__":
    main()
