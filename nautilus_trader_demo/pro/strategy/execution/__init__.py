"""统一目标到模拟撮合器或真实交易端的执行边界。"""

from strategy.execution.contracts import (
    ExecutionBackendKind,
    ExecutionReport,
    ExecutionReportType,
    OrderIntent,
    OrderSide,
    OrderType,
    PositionEffect,
)
from strategy.execution.nautilus import NautilusExecutionAdapter
from strategy.execution.planner import NetTargetOrderPlanner
from strategy.execution.live import (
    BackendExecutionClient,
    NautilusLiveDriverPort,
    NautilusLiveExecutionBackend,
    NautilusTradingNodeDriver,
)
from strategy.execution.ctp import (
    CtpCloseAllocation,
    CtpClosePlanner,
    CtpCommissionRule,
    CtpExecutionAccounting,
    CtpFillResult,
    CtpFuturesHedgingProfile,
    CtpPositionLedger,
    CtpPositionSnapshot,
    CtpSettlementResult,
)
from strategy.execution.ports import (
    ExecutionBackendPort,
    LiveExecutionBackendPort,
    OrderPlannerPort,
    SimExecutionBackendPort,
    VenueSimulationProfilePort,
)
from strategy.execution.simulation import (
    BinanceUsdtFuturesProfile,
    CtpFuturesBasicProfile,
    GenericVenueProfile,
    NautilusSimExecutionBackend,
)

__all__ = [
    "ExecutionBackendKind",
    "ExecutionBackendPort",
    "ExecutionReport",
    "ExecutionReportType",
    "BinanceUsdtFuturesProfile",
    "CtpFuturesBasicProfile",
    "CtpCloseAllocation",
    "CtpClosePlanner",
    "CtpCommissionRule",
    "CtpExecutionAccounting",
    "CtpFillResult",
    "CtpFuturesHedgingProfile",
    "CtpPositionLedger",
    "CtpPositionSnapshot",
    "CtpSettlementResult",
    "GenericVenueProfile",
    "BackendExecutionClient",
    "LiveExecutionBackendPort",
    "NautilusExecutionAdapter",
    "NautilusLiveDriverPort",
    "NautilusLiveExecutionBackend",
    "NautilusTradingNodeDriver",
    "NautilusSimExecutionBackend",
    "OrderIntent",
    "NetTargetOrderPlanner",
    "OrderPlannerPort",
    "OrderSide",
    "OrderType",
    "PositionEffect",
    "SimExecutionBackendPort",
    "VenueSimulationProfilePort",
]
