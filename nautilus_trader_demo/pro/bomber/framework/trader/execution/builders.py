"""构建相互共享仓位／参考价的执行积木；构造不连接、不授权报单。"""

from dataclasses import dataclass
import time

from bomber.framework.trader.portfolio import PositionManager
from bomber.framework.trader.persistence import JsonStateStore, RuntimeStateManager
from .ctp import CtpExecutionAccounting, CtpLimitPlanner, CtpPositionLedger
from .live import ControlledLiveExecutionClient, NautilusLiveExecutionBackend
from .planner import NetTargetOrderPlanner
from .recording import RecordingExecutionClient
from .risk import MarketReferencePriceStore, PreTradeRiskManager
from .simulation import SimulationExecutionClient


@dataclass(frozen=True)
class ExecutionComponents:
    """一个执行账户的公共组件集合；生命周期交给Runtime。"""
    client: object
    positions: PositionManager
    prices: MarketReferencePriceStore
    backend: object = None
    ledger: object = None
    driver: object = None


def build_simulation_execution(backend, *, instrument_limits, positions=None, prices=None):
    positions = PositionManager() if positions is None else positions
    prices = MarketReferencePriceStore() if prices is None else prices
    client = SimulationExecutionClient(backend.backend_id,
        NetTargetOrderPlanner(positions), backend, positions,
        risk_manager=PreTradeRiskManager(backend.backend_id, positions, prices,
            instrument_limits=instrument_limits))
    return ExecutionComponents(client, positions, prices, backend=backend)


def build_ctp_execution(driver, *, instrument_id, trading_day, price_increment,
                        multiplier, risk_limits, session_check, orders=False,
                        limit_offset_ticks=1, positions=None, prices=None, clock_ns=None):
    positions = PositionManager() if positions is None else positions
    prices = MarketReferencePriceStore() if prices is None else prices
    ledger = CtpPositionLedger(trading_day)
    if not orders:
        return ExecutionComponents(RecordingExecutionClient(driver.driver_id),
            positions, prices, ledger=ledger, driver=driver)
    backend = NautilusLiveExecutionBackend(driver.driver_id, driver)
    client = ControlledLiveExecutionClient(driver.driver_id,
        CtpLimitPlanner(ledger, positions, prices, price_increment, limit_offset_ticks),
        backend, positions, PreTradeRiskManager(driver.driver_id, positions, prices,
            instrument_limits={instrument_id: risk_limits}), account_id=driver.account_id,
        demo_environment_check=session_check, max_request_wall_age_ns=120_000_000_000,
        wall_clock_ns=clock_ns or (lambda: time.time_ns()))
    backend.register_report_handler(CtpExecutionAccounting(
        ledger, {instrument_id: multiplier}).on_report)
    return ExecutionComponents(client, positions, prices, backend, ledger, driver)


def attach_ctp_persistence(execution, runner, *, state_file):
    """仅安装写前／回报后钩子，须在最终TD会话和Runner启动前调用。"""
    if execution.backend is None or execution.driver is None:
        raise ValueError("CTP持久化需要受控执行组件")
    if runner.position_manager is not execution.positions:
        raise ValueError("Runner和执行组件须共享PositionManager")
    client_id = execution.client.client_id
    manager = RuntimeStateManager(JsonStateStore(state_file), runner.target_store,
        runner.portfolio_coordinator, execution.positions,
        order_machines={client_id: execution.client.order_state_machine},
        ctp_ledgers={client_id: execution.ledger}, ctp_drivers={client_id: execution.driver})
    manager.enable_ctp_autosave(client_id, execution.client)
    return manager
