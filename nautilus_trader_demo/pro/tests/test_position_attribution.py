"""GAP-01 离线验收：归属、净额、冻结订单计划及同代检查点，不连接柜台。"""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.trader import (
    BackendExecutionClient, ContractAssignment, DynamicExecutionRoute,
    ExecutionReport, ExecutionReportType, ExecutionRoute,
    JsonStateStore, NetTargetOrderPlanner, PositionManager, RuntimeMode,
    RuntimeStateManager, ScheduledContractResolver, StrategyTemplate, UnifiedStrategyRunner,
)
from bomber.framework.trader.execution.attribution import AttributionError
from bomber.framework.trader.execution.contracts import AccountPositionSnapshot
from bomber.framework.trader.execution.simulation.client import SimulationExecutionClient


RB = InstrumentId.from_str("rb2704.SHFE")
NEXT = InstrumentId.from_str("rb2705.SHFE")


class Backend:
    backend_id = "attribution-test"

    def __init__(self):
        self.orders = []
        self.fail = False

    def register_report_handler(self, handler):
        self.handler = handler

    def start(self):
        pass

    def stop(self):
        pass

    def reconcile(self):
        return AccountPositionSnapshot(self.backend_id, 1, 0, {})

    def cancel_strategy(self, strategy_id):
        pass

    def submit_order(self, order):
        if self.fail:
            raise RuntimeError("未发送")
        self.orders.append(order)

    def report(self, index, status, quantity=0, sequence=1, *, reason=None, **metadata):
        order = self.orders[index]
        return ExecutionReport(self.backend_id, f"order-{index}", order.instrument_id,
            status, sequence, filled_quantity=quantity,
            fill_price=3100 if quantity else None, order_side=order.side,
            order_quantity=order.quantity, report_id=f"order-{index}:{sequence}", sequence=sequence,
            reason=reason,
            metadata={**order.metadata, "strategy_id": order.strategy_id,
                "trade_id": f"trade-{index}-{sequence}", **metadata})

    def emit(self, *args, **kwargs):
        report = self.report(*args, **kwargs)
        self.handler(report)
        return report


class Observer(StrategyTemplate):
    def __init__(self, strategy_id):
        super().__init__(strategy_id)
        self.fills = []
        self.updates = []

    def on_order_update(self, event):
        self.updates.append((event.status.value, self.position("leg")))

    def on_fill(self, event):
        self.fills.append((event.quantity, self.position("leg"),
            self.account_position("leg"), event.commission))


class PausedPlanner:
    """第一份目标先保留，第二份提交才按完整组合执行，验证合并订单。"""
    def __init__(self, positions):
        self.planner = NetTargetOrderPlanner(positions)
        self.paused = True

    def plan(self, request):
        return () if self.paused else self.planner.plan(request)


class AttributionTests(unittest.TestCase):
    def setup_runner(self, live=False, paused=False, second_instrument=RB):
        positions, backend = PositionManager(), Backend()
        planner = PausedPlanner(positions) if paused else NetTargetOrderPlanner(positions)
        client_type = BackendExecutionClient if live else SimulationExecutionClient
        client = client_type(backend.backend_id, planner, backend, positions)
        runner = UnifiedStrategyRunner(RuntimeMode.LIVE if live else RuntimeMode.HISTORICAL,
            position_manager=positions)
        runner.add_execution_client(client)
        alpha, beta = Observer("alpha"), Observer("beta")
        for strategy, instrument in ((alpha, RB), (beta, second_instrument)):
            runner.add_strategy(strategy, data_bindings=(),
                execution_routes=(ExecutionRoute("leg", backend.backend_id, instrument),))
        runner.start()
        self.addCleanup(runner.stop)
        return runner, client, backend, alpha, beta

    def test_partial_full_close_and_duplicate_before_callbacks_in_both_clients(self):
        for live in (False, True):
            with self.subTest(live=live):
                runner, client, backend, alpha, _ = self.setup_runner(live=live)
                alpha.set_target("leg", 3, 1)
                backend.emit(0, ExecutionReportType.ACCEPTED)
                partial = backend.emit(0, ExecutionReportType.PARTIALLY_FILLED, 1, 2)
                backend.handler(partial)
                backend.handler(backend.report(0, ExecutionReportType.ACCEPTED))
                backend.emit(0, ExecutionReportType.FILLED, 2, 3)
                self.assertEqual(alpha.fills, [(1, 1, 1, None), (2, 3, 3, None)])
                self.assertEqual(alpha.updates[-1], ("FILLED", Decimal(3)))
                alpha.set_target("leg", 0, 4)
                backend.emit(1, ExecutionReportType.FILLED, 3)
                self.assertEqual(alpha.position("leg"), 0)
                self.assertEqual(alpha.fills[-1][1], 0)
                self.assertEqual(runner.position_manager.unassigned_position(backend.backend_id, RB), 0)
                self.assertFalse(client.report_errors)

    def test_same_direction_merged_order_splits_fill_and_fee(self):
        runner, client, backend, alpha, beta = self.setup_runner(paused=True)
        alpha.set_target("leg", 2, 1)
        client.planner.paused = False
        beta.set_target("leg", 3, 2)
        self.assertEqual(backend.orders[0].quantity, 5)
        backend.emit(0, ExecutionReportType.PARTIALLY_FILLED, 3, commission="6", commission_currency="CNY")
        self.assertEqual(alpha.fills, [(2, 2, 3, 4)])
        self.assertEqual(beta.fills, [(1, 1, 3, 2)])
        backend.emit(0, ExecutionReportType.FILLED, 2, 2)
        self.assertEqual((alpha.position("leg"), beta.position("leg")), (2, 3))
        self.assertEqual(runner.position_manager.unassigned_position(backend.backend_id, RB), 0)
        self.assertFalse(client.report_errors)

    def test_opposite_targets_internal_transfer_and_external_remainder(self):
        runner, client, backend, alpha, beta = self.setup_runner(paused=True)
        alpha.set_target("leg", 2, 1)
        client.planner.paused = False
        beta.set_target("leg", -1, 2)
        self.assertEqual((alpha.position("leg"), beta.position("leg")), (1, -1))
        self.assertEqual(backend.orders[0].quantity, 1)
        backend.emit(0, ExecutionReportType.FILLED, 1)
        self.assertEqual((alpha.position("leg"), beta.position("leg")), (2, -1))
        self.assertEqual(beta.fills, [])  # 内部划转不伪造成交。
        self.assertEqual(runner.position_manager.account_position(backend.backend_id, RB), 1)
        self.assertEqual(len(runner.position_manager.attribution.state()["transfers"]), 1)

    def test_zero_net_transfer_does_not_send_order(self):
        runner, client, backend, alpha, beta = self.setup_runner(paused=True)
        alpha.set_target("leg", 2, 1)
        client.planner.paused = False
        beta.set_target("leg", -2, 2)
        self.assertEqual(backend.orders, [])
        self.assertEqual((alpha.position("leg"), beta.position("leg")), (2, -2))
        self.assertEqual(runner.position_manager.unassigned_position(backend.backend_id, RB), 0)

    def test_changed_target_does_not_reassign_pending_fill(self):
        runner, client, backend, alpha, beta = self.setup_runner()
        alpha.set_target("leg", 2, 1)
        beta.set_target("leg", 1, 2)
        backend.emit(1, ExecutionReportType.FILLED, 1)
        backend.emit(0, ExecutionReportType.FILLED, 2)
        self.assertEqual((alpha.position("leg"), beta.position("leg")), (2, 1))
        self.assertEqual(beta.fills[0][1], 1)
        self.assertFalse(client.report_errors)

    def test_same_strategy_multiple_logical_keys_keep_separate_ownership(self):
        positions, backend = PositionManager(), Backend()
        client = SimulationExecutionClient(backend.backend_id, NetTargetOrderPlanner(positions), backend, positions)
        runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
        runner.add_execution_client(client)
        alpha = Observer("alpha")
        runner.add_strategy(alpha, data_bindings=(), execution_routes=(
            ExecutionRoute("leg", backend.backend_id, RB), ExecutionRoute("second", backend.backend_id, RB)))
        runner.start()
        self.addCleanup(runner.stop)
        alpha.set_targets({"leg": 1, "second": 2}, 1)
        backend.emit(0, ExecutionReportType.FILLED, 3)
        self.assertEqual((alpha.position("leg"), alpha.position("second")), (1, 2))
        alpha.set_targets({"second": 2}, 2)  # REPLACE隐式清掉第一腿。
        backend.emit(1, ExecutionReportType.FILLED, 1)
        self.assertEqual((alpha.position("leg"), alpha.position("second")), (0, 2))
        self.assertEqual(positions.unassigned_position(backend.backend_id, RB), 0)
        self.assertFalse(client.report_errors)

    def test_dynamic_roll_aggregates_actual_old_and_new_contract_ownership(self):
        class RollObserver(StrategyTemplate):
            def __init__(self):
                super().__init__("alpha")
                self.observed = []

            def on_fill(self, event):
                self.observed.append(self.position("leg"))

        positions, backend = PositionManager(), Backend()
        client = SimulationExecutionClient(backend.backend_id, NetTargetOrderPlanner(positions), backend, positions)
        runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
        runner.add_execution_client(client)
        alpha = RollObserver()
        resolver = ScheduledContractResolver((ContractAssignment("leg", RB, 0, 0, 1),
            ContractAssignment("leg", NEXT, 10, 10, 2)))
        runner.add_strategy(alpha, data_bindings=(),
            execution_routes=(DynamicExecutionRoute("leg", backend.backend_id, resolver),))
        runner.start()
        self.addCleanup(runner.stop)
        alpha.set_target("leg", 1, 1)
        backend.emit(0, ExecutionReportType.FILLED, 1)
        runner.refresh_dynamic_routes(10)
        self.assertEqual(len(backend.orders), 1)  # 先进入撤单阶段。
        runner.refresh_dynamic_routes(11)
        backend.emit(1, ExecutionReportType.FILLED, 1)
        runner.refresh_dynamic_routes(12)
        backend.emit(2, ExecutionReportType.FILLED, 1)
        self.assertEqual(alpha.observed, [1, 0, 1])
        self.assertEqual(positions.unassigned_position(backend.backend_id, RB), 0)
        self.assertEqual(positions.unassigned_position(backend.backend_id, NEXT), 0)
        self.assertFalse(client.report_errors)

    def test_different_clients_do_not_net_or_share_attribution(self):
        positions = PositionManager()
        runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
        accounts = []
        for name, quantity in (("account-a", 1), ("account-b", -1)):
            backend = Backend()
            backend.backend_id = name
            client = SimulationExecutionClient(name, NetTargetOrderPlanner(positions), backend, positions)
            runner.add_execution_client(client)
            strategy = Observer(name)
            runner.add_strategy(strategy, data_bindings=(), execution_routes=(ExecutionRoute("leg", name, RB),))
            accounts.append((backend, client, strategy, quantity))
        runner.start()
        self.addCleanup(runner.stop)
        for backend, client, strategy, quantity in accounts:
            strategy.set_target("leg", quantity, 1)
            backend.emit(0, ExecutionReportType.FILLED, 1)
            self.assertEqual(strategy.position("leg"), quantity)
            self.assertEqual(positions.unassigned_position(backend.backend_id, RB), 0)
            self.assertFalse(client.report_errors)

    def test_attribution_updates_even_without_execution_event_subscribers(self):
        runner, client, backend, alpha, _ = self.setup_runner()
        client._execution_handlers.clear()
        alpha.set_target("leg", 1, 1)
        backend.emit(0, ExecutionReportType.FILLED, 1)
        self.assertEqual(alpha.position("leg"), 1)
        self.assertEqual(alpha.fills, [])

    def test_legacy_logical_positions_are_not_guessed_into_real_contracts(self):
        runner, client, backend, alpha, _ = self.setup_runner()
        positions = runner.position_manager
        positions.set_strategy_position("alpha", "leg", 1)
        positions.set_account_position(backend.backend_id, RB, 1)
        state = replace(positions.state(), attribution={})
        positions.restore(state)
        self.assertEqual(positions.unassigned_position(backend.backend_id, RB), 1)
        with self.assertRaises(AttributionError):
            alpha.set_target("leg", 2, 1)
        # 即使账户净仓为零，旧的相抵虚拟仓也不能被重复内部划转。
        positions.set_account_position(backend.backend_id, RB, 0)
        positions.set_strategy_position("beta", "leg", -1)
        with self.assertRaises(AttributionError):
            alpha.set_target("leg", 2, 2)

    def test_cancel_and_reject_release_only_unfilled_reservation(self):
        for live in (False, True):
            for status in (ExecutionReportType.CANCELED, ExecutionReportType.REJECTED):
                with self.subTest(live=live, status=status):
                    runner, client, backend, alpha, _ = self.setup_runner(live=live)
                    alpha.set_target("leg", 3, 1)
                    if status is ExecutionReportType.CANCELED:
                        backend.emit(0, ExecutionReportType.PARTIALLY_FILLED, 1)
                    report = backend.emit(0, status, sequence=2,
                        reason="测试柜台拒单" if status is ExecutionReportType.REJECTED else None)
                    # 先定位拒单／撤单回报本身，避免错误延迟到后续目标重试。
                    self.assertFalse(client.report_errors, client.report_errors)
                    self.assertEqual(alpha.updates[-1][0], status.value)
                    self.assertEqual(runner.position_manager.working_quantity(backend.backend_id, RB), 0)
                    self.assertEqual(alpha.position("leg"),
                        1 if status is ExecutionReportType.CANCELED else 0)
                    if status is ExecutionReportType.REJECTED:
                        self.assertEqual(report.reason, "测试柜台拒单")
                    alpha.set_target("leg", 3, 3)
                    self.assertEqual(backend.orders[1].quantity,
                        2 if status is ExecutionReportType.CANCELED else 3)
                    backend.emit(1, ExecutionReportType.FILLED, backend.orders[1].quantity)
                    self.assertEqual(alpha.position("leg"), 3)
                    self.assertFalse(client.report_errors, client.report_errors)

    def test_unassigned_snapshot_blocks_execution_until_explicit_adoption(self):
        runner, client, backend, alpha, _ = self.setup_runner()
        positions = runner.position_manager
        positions.set_account_position(backend.backend_id, RB, 1)
        self.assertEqual(positions.unassigned_position(backend.backend_id, RB), 1)
        with self.assertRaises(AttributionError):
            alpha.set_target("leg", 2, 1)
        self.assertEqual(backend.orders, [])
        positions.attribution.adopt(backend.backend_id, RB, "alpha", "leg", 1)
        alpha.set_target("leg", 2, 2)
        backend.emit(0, ExecutionReportType.FILLED, 1)
        self.assertEqual(alpha.position("leg"), 2)

    def test_unknown_fill_is_not_guessed_from_strategy_id(self):
        runner, client, backend, alpha, _ = self.setup_runner()
        report = ExecutionReport(backend.backend_id, "external", RB, ExecutionReportType.FILLED,
            1, filled_quantity=1, fill_price=3100, order_side="BUY", order_quantity=1,
            report_id="external:1", metadata={"strategy_id": "alpha", "trade_id": "external-fill"})
        backend.handler(report)
        self.assertEqual(alpha.position("leg"), 0)
        self.assertEqual(alpha.fills, [])
        self.assertEqual(runner.position_manager.unassigned_position(backend.backend_id, RB), 1)
        with self.assertRaises(RuntimeError):
            alpha.set_target("leg", 2, 2)

    def test_conflicting_association_updates_account_but_closes_gate(self):
        runner, client, backend, alpha, beta = self.setup_runner(live=True)
        alpha.set_target("leg", 1, 1)
        backend.emit(0, ExecutionReportType.FILLED, 1, strategy_id="beta")
        self.assertEqual((alpha.position("leg"), beta.position("leg")), (0, 0))
        self.assertEqual(runner.position_manager.unassigned_position(backend.backend_id, RB), 1)
        self.assertFalse(client.is_reconciled)
        self.assertTrue(client.report_errors)

    def test_unsent_order_releases_reservation(self):
        runner, client, backend, alpha, _ = self.setup_runner()
        backend.fail = True
        with self.assertRaisesRegex(RuntimeError, "未发送"):
            alpha.set_target("leg", 1, 1)
        self.assertEqual(runner.position_manager.working_quantity(backend.backend_id, RB), 0)
        backend.fail = False
        alpha.set_target("leg", 1, 2)
        backend.emit(0, ExecutionReportType.FILLED, 1)
        self.assertEqual(alpha.position("leg"), 1)

    def test_different_instruments_never_net(self):
        runner, client, backend, alpha, beta = self.setup_runner(second_instrument=NEXT)
        alpha.set_target("leg", 1, 1)
        beta.set_target("leg", -1, 2)
        self.assertEqual(len(backend.orders), 2)
        backend.emit(0, ExecutionReportType.FILLED, 1)
        backend.emit(1, ExecutionReportType.FILLED, 1)
        self.assertEqual((alpha.position("leg"), beta.position("leg")), (1, -1))
        self.assertEqual(runner.position_manager.unassigned_position(backend.backend_id, RB), 0)
        self.assertEqual(runner.position_manager.unassigned_position(backend.backend_id, NEXT), 0)

    def test_restart_preserves_association_and_duplicate_fill_keys(self):
        runner, client, backend, alpha, _ = self.setup_runner()
        alpha.set_target("leg", 3, 1)
        partial = backend.emit(0, ExecutionReportType.PARTIALLY_FILLED, 1)
        with TemporaryDirectory() as root:
            manager = RuntimeStateManager(JsonStateStore(Path(root) / "state.json"),
                runner.target_store, runner.portfolio_coordinator, runner.position_manager,
                order_machines={backend.backend_id: client.order_state_machine})
            manager.save()
            restored, restored_client, restored_backend, restored_alpha, _ = self.setup_runner()
            recovery = RuntimeStateManager(JsonStateStore(Path(root) / "state.json"),
                restored.target_store, restored.portfolio_coordinator, restored.position_manager,
                order_machines={backend.backend_id: restored_client.order_state_machine})
            recovery.restore()
            self.assertTrue(restored.position_manager.is_recovery_required(backend.backend_id))
            # 只恢复同代检查点并重放回报；通用在线重启入口仍由GAP-06验收。
            restored_backend.orders = list(backend.orders)
            restored_backend.handler(partial)
            self.assertEqual(restored_alpha.position("leg"), 1)
            final = restored_backend.report(0, ExecutionReportType.FILLED, 2, 2)
            # 已在首次回报/检查点绑定订单ID，后续可使用关联而无需重复携带token。
            final = replace(final, metadata={k: v for k, v in final.metadata.items() if k != "allocation_token"})
            restored_backend.handler(final)
            self.assertEqual(restored_alpha.position("leg"), 3)
            self.assertEqual(restored_alpha.fills[0][1], 3)
            self.assertEqual(restored.position_manager.unassigned_position(backend.backend_id, RB), 0)
            self.assertFalse(restored_client.report_errors)

    def test_mismatched_checkpoint_is_rejected_before_restore(self):
        runner, client, backend, alpha, _ = self.setup_runner()
        alpha.set_target("leg", 3, 1)
        backend.emit(0, ExecutionReportType.PARTIALLY_FILLED, 1)
        state = runner.position_manager.attribution.state()
        next(iter(state["orders"].values()))["legs"][0]["filled"] = "0"
        with self.assertRaises(AttributionError):
            runner.position_manager.attribution.validate_checkpoints(state,
                {backend.backend_id: client.order_state_machine.checkpoints()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
