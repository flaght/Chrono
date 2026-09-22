"""P3只读账户查询转换及心跳闭闸测试；不连接Binance。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from strategy import NautilusReportedAccountReader, NautilusTradingNodeDriver
from tests.run_p3_controlled_live import build_client


@dataclass
class FakeMoney:
    value: str

    def as_decimal(self) -> Decimal:
        return Decimal(self.value)


@dataclass
class FakeCurrency:
    code: str


@dataclass
class FakeBalance:
    currency: FakeCurrency
    total: FakeMoney
    free: FakeMoney


@dataclass
class FakeEvent:
    id: str
    is_reported: bool
    info: dict
    balances: list[FakeBalance]
    ts_event: int = 100


@dataclass
class FakeAccount:
    id: str = "BINANCE-TEST"
    events: list[FakeEvent] = field(default_factory=list)


def test1_fresh_http_account_state() -> None:
    """只接受本次查询后新产生且带HTTP独有字段的交易所回报。"""
    account = FakeAccount(events=[FakeEvent("old", True, {}, [])])
    calls = []

    def query(native_id):
        calls.append(native_id)
        balance = FakeBalance(FakeCurrency("USDT"), FakeMoney("1000"), FakeMoney("800"))
        account.events.append(FakeEvent("ws", True, {}, [balance]))
        account.events.append(FakeEvent(
            "http", True,
            {"total_margin_balance": "1050", "total_initial_margin": "200",
             "available_balance": "790"},
            [balance],
        ))

    reader = NautilusReportedAccountReader(
        "demo-client", "demo-account", lambda: account, query,
        required_info_keys=("total_margin_balance", "available_balance"),
        timeout_seconds=0.1, poll_seconds=0.001,
    )
    state = reader.read()
    assert calls == ["BINANCE-TEST"]
    assert state.revision == 1
    assert state.balances["USDT"].total == 1000
    assert state.balances["USDT"].equity == 1050
    assert state.balances["USDT"].available == 790
    assert state.balances["USDT"].margin_used == 200
    try:
        reader.read()
    except TimeoutError:
        pass
    else:
        raise AssertionError("旧HTTP事件不能冒充下一次查询结果")
    print("P3d1通过：只读QueryAccount必须取得新的HTTP权威回报，忽略WebSocket/旧缓存")


def test2_heartbeat_closes_gate() -> None:
    """资金心跳失败不得保留已授权状态。"""
    client, driver, positions, _ = build_client()
    client.start()
    initial_revision = client.account_state.revision
    assert client.refresh_account_state().revision > initial_revision
    client.arm_demo(client.DEMO_CONFIRMATION)
    driver.fail_account_query = True
    try:
        client.refresh_account_state()
    except RuntimeError as error:
        assert "资金查询失败" in str(error)
    else:
        raise AssertionError("资金查询失败必须闭闸")
    assert not client.is_armed and not client.is_reconciled
    assert positions.is_recovery_required("demo-client")
    client.stop()
    print("P3d2通过：周期性资金查询失败时撤销DEMO授权并要求完整恢复")


def test3_query_runs_on_node_loop() -> None:
    """只读查询必须投递到TradingNode自己的事件循环。"""
    calls = []

    class FakeLoop:
        def is_running(self):
            return True

        def call_soon_threadsafe(self, callback, account_id):
            calls.append("scheduled")
            callback(account_id)

    class FakeNode:
        def get_event_loop(self):
            return FakeLoop()

    class FakeGateway:
        def query_account(self, account_id):
            calls.append(account_id)

    driver = NautilusTradingNodeDriver("demo-client", FakeNode())
    driver._started = True
    driver._gateway = FakeGateway()
    driver.query_account("BINANCE-TEST")
    assert calls == ["scheduled", "BINANCE-TEST"]
    print("P3d3通过：QueryAccount仅在TradingNode事件循环上调度")


def main() -> None:
    test1_fresh_http_account_state()
    test2_heartbeat_closes_gate()
    test3_query_runs_on_node_loop()


if __name__ == "__main__":
    main()
