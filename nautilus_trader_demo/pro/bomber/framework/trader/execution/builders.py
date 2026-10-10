"""构建相互共享仓位／参考价的执行积木；构造不连接、不授权报单。"""

from dataclasses import dataclass
from decimal import Decimal
import time

from bomber.framework.trader.portfolio import PositionManager
from bomber.framework.trader.persistence import JsonStateStore, RuntimeStateManager
from .ctp import CtpExecutionAccounting, CtpLimitPlanner, CtpPositionLedger
from .live import ControlledLiveExecutionClient, NautilusLiveExecutionBackend
from .planner import NetTargetOrderPlanner
from .recording import RecordingExecutionClient
from .risk import MarketReferencePriceStore, PreTradeRiskManager
from .simulation import SimulationExecutionClient
from .contracts import ExecutionReportType


class _CtpPortfolioBackend(NautilusLiveExecutionBackend):
    instrument_scope = None

    def submit_order(self, order):
        if self.instrument_scope is not None and not self.instrument_scope.contains(order.instrument_id):
            raise RuntimeError("订单合约不在本次执行范围内")
        # 每腿发送前再次检查；同步拒单/断线后不能继续本批其余腿。
        if not self.submission_check():
            raise RuntimeError("多合约会话已失效或发生拒单，停止后续腿")
        return super().submit_order(order)


class _CtpPortfolioClient(ControlledLiveExecutionClient):
    portfolio_failure = None

    def _on_report(self, report):
        with self._submit_lock:
            super()._on_report(report)
            if report.report_type is ExecutionReportType.REJECTED or self.report_errors:
                self.portfolio_failure = "组合委托拒单或执行回报冲突，须核对已成交腿"
                self.disarm("portfolio_report_failure")
                self.position_manager.mark_recovery_required(self.client_id)


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


def attach_ctp_persistence(execution, runner, *, state_file, state_components=None):
    """仅安装写前／回报后钩子，须在最终TD会话和Runner启动前调用。"""
    if execution.backend is None or execution.driver is None:
        raise ValueError("CTP持久化需要受控执行组件")
    if runner.position_manager is not execution.positions:
        raise ValueError("Runner和执行组件须共享PositionManager")
    client_id = execution.client.client_id
    manager = RuntimeStateManager(JsonStateStore(state_file), runner.target_store,
        runner.portfolio_coordinator, execution.positions,
        order_machines={client_id: execution.client.order_state_machine},
        ctp_ledgers={client_id: execution.ledger}, ctp_drivers={client_id: execution.driver},
        state_components=state_components)
    manager.enable_ctp_autosave(client_id, execution.client)
    return manager


def build_ctp_portfolio_execution(driver, *, trading_day, price_increments,
                                  multipliers, instrument_limits, session_check,
                                  orders=False, limit_offset_ticks=1, positions=None,
                                  prices=None, clock_ns=None, instrument_scope=None):
    """固定多合约装配；可显式限定管理范围，账户快照仍保留全部仓位。"""
    keys = set(price_increments)
    if not keys or keys != set(multipliers) or keys != set(instrument_limits):
        raise ValueError("多合约跳价、乘数与风控限额须覆盖同一非空集合")
    if instrument_scope is not None and instrument_scope.instruments != frozenset(str(i) for i in keys):
        raise ValueError("执行合约范围须与多合约配置一致")
    for key in keys:
        multiplier = Decimal(str(multipliers[key]))
        if not multiplier.is_finite() or multiplier <= 0:
            raise ValueError(f"合约乘数须为有限正数: {key}")
        if multiplier != instrument_limits[key].contract_multiplier:
            raise ValueError(f"风控与成交记账乘数不一致: {key}")
    positions = PositionManager() if positions is None else positions
    prices = MarketReferencePriceStore() if prices is None else prices
    ledger = CtpPositionLedger(trading_day)
    planner = CtpLimitPlanner(ledger, positions, prices, price_increments, limit_offset_ticks)
    if not orders:
        return ExecutionComponents(RecordingExecutionClient(driver.driver_id),
            positions, prices, ledger=ledger, driver=driver)
    backend = _CtpPortfolioBackend(driver.driver_id, driver)
    backend.instrument_scope = instrument_scope
    client = _CtpPortfolioClient(driver.driver_id, planner, backend, positions,
        PreTradeRiskManager(driver.driver_id, positions, prices, instrument_limits=instrument_limits),
        account_id=driver.account_id, demo_environment_check=session_check,
        max_request_wall_age_ns=120_000_000_000,
        wall_clock_ns=clock_ns or (lambda: time.time_ns()))
    backend.submission_check = lambda: (session_check() and client.is_reconciled and not client.portfolio_failure
                                       and not client.report_errors)
    backend.register_report_handler(CtpExecutionAccounting(ledger, multipliers).on_report)
    return ExecutionComponents(client, positions, prices, backend, ledger, driver)
