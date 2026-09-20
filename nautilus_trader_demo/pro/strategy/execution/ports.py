"""可替换执行后端、订单规划器和市场模拟 Profile 协议。"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from strategy.contracts import ExecutionRequest
from strategy.execution.contracts import ExecutionBackendKind, ExecutionReport, OrderIntent


@runtime_checkable
class ExecutionBackendPort(Protocol):
    """模拟撮合与实盘执行共同遵守的最小订单执行边界。"""

    backend_id: str
    kind: ExecutionBackendKind

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def submit_order(self, order: OrderIntent) -> None: ...

    def register_report_handler(
        self,
        handler: Callable[[ExecutionReport], None],
    ) -> None: ...

    def cancel_strategy(self, strategy_id: str) -> None: ...


@runtime_checkable
class SimExecutionBackendPort(ExecutionBackendPort, Protocol):
    """需要消费标准行情并能输出回测结果的模拟执行后端。"""

    def process_market_event(self, event: Any) -> Sequence[ExecutionReport]: ...

    def result(self) -> Any: ...


@runtime_checkable
class LiveExecutionBackendPort(ExecutionBackendPort, Protocol):
    """连接真实或模拟柜台、支持权威状态同步的在线执行后端。"""

    def reconcile(self) -> None: ...


@runtime_checkable
class OrderPlannerPort(Protocol):
    """把账户目标与当前状态之差规划为一个或多个订单意图。"""

    planner_id: str

    def plan(self, request: ExecutionRequest) -> Sequence[OrderIntent]: ...


@runtime_checkable
class VenueSimulationProfilePort(Protocol):
    """向通用模拟内核提供交易场所规则和可插拔模型配置。"""

    profile_id: str
    venue: str

    def build_backend_config(self) -> Mapping[str, Any]: ...
