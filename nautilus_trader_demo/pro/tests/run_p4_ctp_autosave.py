"""P4-CTP6：CTP报单写前检查点和柜台回报后的自动保存。"""

from __future__ import annotations

from pathlib import Path
import tempfile

from market.basic.base import InstrumentId
from strategy.contracts import ExecutionRequest
from strategy.execution.ctp.native_driver import CtpNativeTraderDriver
from strategy.execution.live.backend import NautilusLiveExecutionBackend
from strategy.execution.live.client import BackendExecutionClient
from strategy.persistence import JsonStateStore, RuntimeStateManager
from strategy.portfolio import PortfolioCoordinator, PositionManager, TargetStore
from run_p4_ctp_driver import FakeTraderTransport, _intent, _raw


RB = InstrumentId.from_str("rb2704.SHFE")


class FixedPlanner:
    def plan(self, request):
        return (_intent(),)


class FailingTransport(FakeTraderTransport):
    def send_order(self, fields):
        self.sent.append(dict(fields))
        raise RuntimeError("CTP请求同步失败")


class CallbackThenFailTransport(FakeTraderTransport):
    def send_order(self, fields):
        self.sent.append(dict(fields))
        self.on_order(_raw(fields["OrderRef"], OrderStatus="3"))
        raise RuntimeError("请求返回异常但订单已受理")


def _build(path: Path, transport):
    positions = PositionManager()
    driver = CtpNativeTraderDriver(
        "ctp-demo", "demo-account", transport,
        enable_test_orders=True, disconnect_handler=lambda reason: None,
    )
    backend = NautilusLiveExecutionBackend("ctp-demo", driver)
    client = BackendExecutionClient(
        "ctp-demo", FixedPlanner(), backend, positions,
        account_id="demo-account",
    )
    manager = RuntimeStateManager(
        JsonStateStore(path), TargetStore(), PortfolioCoordinator(), positions,
        order_machines={"ctp-demo": client.order_state_machine},
        ctp_drivers={"ctp-demo": driver},
    )
    manager.enable_ctp_autosave("ctp-demo", client)
    return client, driver, manager


def _request():
    return ExecutionRequest(
        strategy_id="alpha", revision=1, client_id="ctp-demo",
        ts_event=100, targets={RB: 0}, execution_policy="DIRECT",
    )


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.json"
        transport = FakeTraderTransport()
        client, driver, manager = _build(path, transport)
        client.start()
        client.submit_targets(_request())
        assert len(transport.sent) == 1
        assert manager.generation == 1
        state = client.order_state("CTP-9999-demo-20260922-8")
        assert state is not None and state.status.value == "PENDING"
        assert JsonStateStore(path).load().payload["ctp_drivers"]["ctp-demo"]["orders"]
        print("P4-CTP6a通过：请求发送前待确认订单及CTP关联已同代落盘")

        transport.on_order(_raw("8", OrderStatus="3"))
        transport.on_trade(_raw("8", TradeID="T1", Volume=1, Price=3125))
        assert manager.generation == 3
        transport.on_trade(_raw("8", TradeID="T1", Volume=1, Price=3125))
        assert manager.generation == 3
        transport.on_trade(_raw("8", TradeID="T2", Volume=1, Price=3126))
        assert manager.generation == 4
        assert not JsonStateStore(path).load().payload["ctp_drivers"]["ctp-demo"]["orders"]
        assert client.order_state("CTP-9999-demo-20260922-8").status.value == "FILLED"
        client.stop()
        print("P4-CTP6b通过：受理、部分及全部成交后自动保存，重复成交不推进版本")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "failed-send.json"
        transport = FailingTransport()
        client, driver, manager = _build(path, transport)
        client.start()
        try:
            client.submit_targets(_request())
        except RuntimeError as error:
            assert "同步失败" in str(error)
        else:
            raise AssertionError("同步报单失败必须向调用方报错")
        assert manager.generation == 2
        assert client.order_state("CTP-9999-demo-20260922-8") is None
        assert not JsonStateStore(path).load().payload["ctp_drivers"]["ctp-demo"]["orders"]
        assert client.position_manager.working_quantity("ctp-demo", RB) == 0
        client.stop()
        print("P4-CTP6c通过：同步请求失败撤回待确认订单和在途量并再次落盘")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "write-failed.json"
        transport = FakeTraderTransport()
        client, driver, manager = _build(path, transport)
        client.start()
        JsonStateStore(path).save({"foreign": True}, expected_generation=0)
        try:
            client.submit_targets(_request())
        except Exception:
            pass
        else:
            raise AssertionError("写前检查点失败必须拒绝报单")
        assert not transport.sent and not client.is_reconciled
        client.stop()
        print("P4-CTP6d通过：写前检查点失败不触达柜台并关闭提交闸门")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ambiguous-send.json"
        transport = CallbackThenFailTransport()
        client, driver, manager = _build(path, transport)
        client.start()
        try:
            client.submit_targets(_request())
        except RuntimeError as error:
            assert "重新对账" in str(error)
        else:
            raise AssertionError("已有柜台回报的发送异常必须保持不确定状态")
        assert manager.generation == 2
        assert not client.is_reconciled
        assert client.position_manager.is_recovery_required("ctp-demo")
        assert client.position_manager.working_quantity("ctp-demo", RB) == -2
        client.stop()
        print("P4-CTP6e通过：柜台已受理但调用异常时保留在途量并要求权威恢复")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "report-write-failed.json"
        transport = FakeTraderTransport()
        client, driver, manager = _build(path, transport)
        client.start()
        client.submit_targets(_request())
        state = JsonStateStore(path).load()
        JsonStateStore(path).save(state.payload, expected_generation=state.generation)
        try:
            transport.on_order(_raw("8", OrderStatus="3"))
        except Exception:
            pass
        else:
            raise AssertionError("回报后保存失败必须向上报告")
        assert not driver._connected
        assert manager.generation == 1
        assert client.order_state("CTP-9999-demo-20260922-8").status.value == "ACCEPTED"
        client.stop()
        print("P4-CTP6f通过：回报后保存失败立即关闭CTP Driver，旧快照待权威核对")


if __name__ == "__main__":
    main()
