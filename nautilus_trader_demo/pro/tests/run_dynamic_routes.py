"""第三类阶段1-3：因果选约、换月状态机和统一Runner动态路由。"""

from __future__ import annotations

from decimal import Decimal

from market.basic.base import InstrumentId
from trader import (
    ContractAssignment,
    ContractUnavailable,
    DynamicExecutionRoute,
    PositionManager,
    RecordingExecutionClient,
    ExecutionRoute,
    RollPhase,
    RuntimeMode,
    SafeRollCoordinator,
    ScheduledContractResolver,
    StrategyTemplate,
    TargetPortfolio,
    UnifiedStrategyRunner,
)


OLD = InstrumentId.from_str("rb2704.SHFE")
NEW = InstrumentId.from_str("rb2705.SHFE")
OLD_ASSIGNMENT = ContractAssignment("rb_main", OLD, 0, 0, 1)
NEW_ASSIGNMENT = ContractAssignment("rb_main", NEW, 100, 110, 2)


class _NoSignalStrategy(StrategyTemplate):
    """信号由测试显式提交；策略实例只用于检验统一Runner装配。"""


class _RecordingClient(RecordingExecutionClient):
    def __init__(self, client_id: str) -> None:
        super().__init__(client_id)
        self.canceled: list[str] = []

    def cancel_strategy(self, strategy_id: str) -> None:
        self.canceled.append(strategy_id)


def test1_causal_resolver() -> None:
    resolver = ScheduledContractResolver((OLD_ASSIGNMENT, NEW_ASSIGNMENT))
    assert resolver.resolve("rb_main", 99).instrument_id == OLD
    try:
        resolver.resolve("rb_main", 105)
    except ContractUnavailable:
        pass
    else:
        raise AssertionError("新版本尚未可用时不能沿用旧合约或偷看新合约")
    assert resolver.resolve("rb_main", 110).instrument_id == NEW
    print("第三类阶段1通过：合约表按as-of可用时间查询，未来版本fail-closed")


def test2_roll_state_machine() -> None:
    roll = SafeRollCoordinator()
    assert roll.step("alpha", OLD_ASSIGNMENT, 2).targets == {OLD: Decimal(2)}
    changed = roll.step("alpha", NEW_ASSIGNMENT, 2, old_position=2, old_working=1)
    assert changed.cancel_strategy_orders and changed.targets is None
    assert roll.step("alpha", NEW_ASSIGNMENT, -3, old_position=2, old_working=1).targets is None
    closing = roll.step("alpha", NEW_ASSIGNMENT, -3, old_position=2, old_working=0)
    assert closing.targets == {OLD: Decimal(0)} and closing.phase is RollPhase.CLOSING
    assert roll.step("alpha", NEW_ASSIGNMENT, -3, old_position=2, old_working=1).targets is None
    recovered = SafeRollCoordinator()
    recovered.restore(roll.snapshot())
    opened = recovered.step("alpha", NEW_ASSIGNMENT, -3, old_position=0, old_working=0)
    assert opened.targets == {OLD: Decimal(0), NEW: Decimal(-3)}
    assert recovered.state("alpha", "rb_main").active.instrument_id == NEW
    print("第三类阶段2通过：撤单、平旧、确认归零、最新信号重开及状态恢复")


def test3_runner_assembly() -> None:
    resolver = ScheduledContractResolver((OLD_ASSIGNMENT, NEW_ASSIGNMENT))
    positions = PositionManager()
    client = _RecordingClient("roll-account")
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_execution_client(client)
    runner.add_strategy(
        _NoSignalStrategy("alpha"),
        data_bindings=(),
        execution_routes=(DynamicExecutionRoute("rb_main", client.client_id, resolver),),
    )
    runner.start()
    try:
        runner.submit(TargetPortfolio("alpha", 1, 50, {"rb_main": 2}))
        assert client.requests[-1].targets[OLD] == 2
        positions.set_account_position(client.client_id, OLD, 2)
        positions.set_working_quantity(client.client_id, OLD, 1)
        try:
            runner.refresh_dynamic_routes(105)
        except ContractUnavailable:
            pass
        else:
            raise AssertionError("不可用的新选约不能触发换月")
        runner.refresh_dynamic_routes(110)
        assert client.canceled == ["alpha"] and len(client.requests) == 1
        runner.refresh_dynamic_routes(111)
        assert len(client.requests) == 1
        positions.set_working_quantity(client.client_id, OLD, 0)
        runner.refresh_dynamic_routes(112)
        assert client.requests[-1].targets[OLD] == 0
        assert NEW not in client.requests[-1].targets
        runner.submit(TargetPortfolio("alpha", 2, 113, {"rb_main": -3}))
        assert all(request.targets.get(NEW, Decimal(0)) == 0 for request in client.requests)
        positions.set_account_position(client.client_id, OLD, 0)
        runner.refresh_dynamic_routes(114)
        assert client.requests[-1].targets[OLD] == 0
        assert client.requests[-1].targets[NEW] == -3
        assert runner.target_store.get("alpha").targets["rb_main"] == -3
    finally:
        runner.stop()
    print("第三类阶段3通过：Runner保持逻辑目标，动态路由只在旧仓归零后开新仓")


def test4_safety_boundaries() -> None:
    resolver = ScheduledContractResolver((OLD_ASSIGNMENT, NEW_ASSIGNMENT))
    positions = PositionManager()
    client = _RecordingClient("exclusive-account")
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_execution_client(client)
    runner.add_strategy(
        _NoSignalStrategy("alpha"),
        data_bindings=(),
        execution_routes=(DynamicExecutionRoute("rb_main", client.client_id, resolver),),
    )
    try:
        runner.add_strategy(
            _NoSignalStrategy("unsafe-multileg"),
            data_bindings=(),
            execution_routes=(
                DynamicExecutionRoute("future", "other-account", resolver),
                DynamicExecutionRoute("call", "other-account", resolver),
            ),
        )
    except ValueError as error:
        assert "单逻辑目标" in str(error)
    else:
        raise AssertionError("未实现多腿原子协调前不得开放多条动态路由")
    try:
        runner.add_strategy(
            _NoSignalStrategy("other"),
            data_bindings=(),
            execution_routes=(ExecutionRoute("other_key", client.client_id, OLD),),
        )
    except ValueError as error:
        assert "独占" in str(error)
    else:
        raise AssertionError("共享账户不能绕过动态换月的旧仓确认")
    runner.start()
    try:
        positions.set_account_position(client.client_id, OLD, 1)
        try:
            runner.submit(TargetPortfolio("alpha", 1, 50, {"rb_main": 2}))
        except RuntimeError:
            pass
        else:
            raise AssertionError("已有未知仓位时不得当作空账户首次启动")
        assert runner.target_store.get("alpha") is None
        assert not client.requests
        positions.set_account_position(client.client_id, OLD, 0)
        positions.complete_working_recovery(client.client_id, {})
        runner.submit(TargetPortfolio("alpha", 1, 50, {"rb_main": 2}))
        assert client.requests[-1].targets[OLD] == 2
    finally:
        runner.stop()
    live_runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    try:
        live_runner.add_strategy(
            _NoSignalStrategy("live"),
            data_bindings=(),
            execution_routes=(DynamicExecutionRoute("rb_main", "live-account", resolver),),
        )
    except ValueError as error:
        assert "HISTORICAL" in str(error)
    else:
        raise AssertionError("未经生产级恢复验收的动态换月不得装配到LIVE")
    print("第三类阶段3安全边界通过：账户独占、未知实仓拒绝及LIVE禁用")


if __name__ == "__main__":
    test1_causal_resolver()
    test2_roll_state_machine()
    test3_runner_assembly()
    test4_safety_boundaries()
