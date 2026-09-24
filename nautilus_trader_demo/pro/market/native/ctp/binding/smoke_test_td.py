"""No-network smoke check for the project-owned TraderApi extension."""

from bomber_ctp_td import TdApi


def main() -> None:
    api = TdApi()
    for name in (
        "createFtdcTraderApi", "registerFront", "init", "exit",
        "reqAuthenticate", "reqUserLogin", "reqSettlementInfoConfirm",
        "reqQryInvestorPosition", "reqQryTradingAccount", "reqQryOrder",
        "reqOrderInsert", "reqOrderAction",
    ):
        if not callable(getattr(api, name, None)):
            raise AssertionError(f"TraderApi method missing: {name}")
    print("bomber_ctp_td methods OK (no network, no orders)")


if __name__ == "__main__":
    main()
