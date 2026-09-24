"""P4-CTP5：跨重启OrderRef归属与柜台活动订单双向核验。"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import tempfile

from trader.execution.ctp.native_driver import CtpNativeTraderDriver
from trader.execution.events import ActiveOrder, ActiveOrderSnapshot
from trader.execution.order_state import OrderReportStateMachine
from trader.persistence import JsonStateStore, RuntimeStateManager, StatePersistenceError
from trader.portfolio import PortfolioCoordinator, PositionManager, TargetStore
from run_p4_ctp_driver import FakeTraderTransport, _intent, _raw


class RecoveryTransport(FakeTraderTransport):
    def __init__(self, orders=()):
        super().__init__()
        self.active_orders = orders

    def query_active_orders(self):
        return ActiveOrderSnapshot(
            "ctp-demo", "demo-account", 1, 100, tuple(self.active_orders),
        )


def _manager(path: Path, driver, machine):
    return RuntimeStateManager(
        JsonStateStore(path), TargetStore(), PortfolioCoordinator(),
        PositionManager(), order_machines={"ctp-demo": machine},
        ctp_drivers={"ctp-demo": driver},
    )


def _active(*, remaining=1, instrument="rb2704.SHFE"):
    return ActiveOrder(
        "CTP-9999-demo-20260922-8", instrument, "SELL", 2,
        2 - remaining, remaining,
    )


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ctp-state.json"
        original_transport = RecoveryTransport((_active(),))
        original_machine = OrderReportStateMachine("ctp-demo")
        original = CtpNativeTraderDriver(
            "ctp-demo", "demo-account", original_transport,
            enable_test_orders=True, disconnect_handler=lambda reason: None,
        )
        original.start(original_machine.apply)
        original.submit_order(_intent())
        original_transport.on_order(_raw("8", OrderStatus="3"))
        original_transport.on_trade(_raw("8", TradeID="T1", Volume=1, Price=3125))
        try:
            _manager(path, original, OrderReportStateMachine("ctp-demo")).save()
        except StatePersistenceError:
            pass
        else:
            raise AssertionError("Driver与订单状态机不同步不能持久化")
        assert not path.exists()
        _manager(path, original, original_machine).save()
        original.stop()

        restored_transport = RecoveryTransport((_active(),))
        restored_machine = OrderReportStateMachine("ctp-demo")
        restored = CtpNativeTraderDriver(
            "ctp-demo", "demo-account", restored_transport,
            enable_test_orders=True, disconnect_handler=lambda reason: None,
        )
        manager = _manager(path, restored, restored_machine)
        manager.restore()
        restored.start(restored_machine.apply)
        try:
            restored.submit_order(_intent())
        except RuntimeError as error:
            assert "尚未" in str(error)
        else:
            raise AssertionError("权威快照核对前不能继续提交CTP订单")
        restored.reconcile_active_orders()
        assert restored.checkpoint().orders[0].intent.strategy_id == "alpha"
        assert restored_machine.state("CTP-9999-demo-20260922-8").filled_quantity == 1
        restored_transport.on_trade(_raw("8", TradeID="T1", Volume=1, Price=3125))
        assert restored_machine.state("CTP-9999-demo-20260922-8").filled_quantity == 1
        restored_transport.on_trade(_raw("8", TradeID="T2", Volume=1, Price=3126))
        assert restored_machine.state("CTP-9999-demo-20260922-8").filled_quantity == 2
        restored.stop()
        print("P4-CTP5a通过：同代持久化、权威活动订单核对与跨重启成交去重正常")

        unknown = ActiveOrder(
            "CTP-9999-demo-20260922-9", "rb2704.SHFE", "SELL", 1, 0, 1,
        )
        for bad_orders in (
            (), (_active(instrument="rb2705.SHFE"),),
            (_active(remaining=2),), (_active(), unknown),
        ):
            transport = RecoveryTransport(bad_orders)
            driver = CtpNativeTraderDriver(
                "ctp-demo", "demo-account", transport,
                enable_test_orders=True, disconnect_handler=lambda reason: None,
            )
            machine = OrderReportStateMachine("ctp-demo")
            _manager(path, driver, machine).restore()
            driver.start(machine.apply)
            try:
                driver.reconcile_active_orders()
            except RuntimeError:
                pass
            else:
                raise AssertionError("柜台活动订单与本地检查点不一致必须闭闸")
            assert not driver._connected and not driver._orders
            driver.stop()
        print("P4-CTP5b通过：订单消失、未知订单、标的或成交量冲突均不恢复归属")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "empty-ctp-state.json"
        empty_driver = CtpNativeTraderDriver(
            "ctp-demo", "demo-account", RecoveryTransport(),
            disconnect_handler=lambda reason: None,
        )
        empty_driver.start(lambda report: None)
        _manager(path, empty_driver, OrderReportStateMachine("ctp-demo")).save()
        empty_driver.stop()
        restarted = CtpNativeTraderDriver(
            "ctp-demo", "demo-account", RecoveryTransport((unknown,)),
            disconnect_handler=lambda reason: None,
        )
        manager = _manager(path, restarted, OrderReportStateMachine("ctp-demo"))
        manager.restore()
        assert manager.position_manager.is_recovery_required("ctp-demo")
        restarted.start(lambda report: None)
        try:
            restarted.reconcile_active_orders()
        except RuntimeError:
            pass
        else:
            raise AssertionError("本地空订单也必须拦截柜台未知活动订单")
        restarted.stop()
        print("P4-CTP5c通过：本地空检查点仍强制查询柜台，未知活动订单闭闸")


if __name__ == "__main__":
    main()
