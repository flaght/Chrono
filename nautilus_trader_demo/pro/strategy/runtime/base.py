"""运行时端口：定义时间、事件推进和生命周期边界。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from strategy.contracts import RuntimeMode


class RunnerLifecyclePort(Protocol):
    """实时适配器依赖的最小Runner接口。"""

    mode: RuntimeMode

    def start(self) -> None: ...

    def stop(self) -> None: ...


class HistoricalRunnerPort(RunnerLifecyclePort, Protocol):
    """轻量回放适配器额外需要的Runner接口。"""

    def run_replay(self) -> Any: ...


@runtime_checkable
class RuntimePort(Protocol):
    """所有实时、回放和正式回测运行时的最小公共接口。"""

    runtime_id: str
    mode: RuntimeMode

    def start(self) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class HistoricalRuntimePort(RuntimePort, Protocol):
    """能够完成一次有限历史运行的公共接口。"""

    def run(self) -> Any: ...


@runtime_checkable
class BacktestRuntimePort(HistoricalRuntimePort, Protocol):
    """正式回测运行时协议。

    与轻量回放不同，正式回测实现必须拥有唯一模拟时钟，并负责撮合、账户、
    仓位和结果统计。第一阶段只固定协议，Nautilus实现将在后续阶段加入。
    """

    engine_name: str
