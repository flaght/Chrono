"""策略运行框架使用的可插拔端口协议。"""

from __future__ import annotations

from decimal import Decimal
from typing import Callable, Protocol, runtime_checkable

from strategy.contracts import ExecutionRequest
from strategy.execution.events import FillEvent, OrderUpdateEvent


class ExecutionClientPort(Protocol):
    """NT、Bomber、vn.py 和模拟交易客户端共同实现的目标接口。"""

    client_id: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def submit_targets(self, request: ExecutionRequest) -> None: ...

    def cancel_strategy(self, strategy_id: str) -> None: ...


@runtime_checkable
class ExecutionEventSourcePort(Protocol):
    """可选双向能力；Recording客户端无需伪造订单或成交。"""

    def register_execution_event_handler(
        self, handler: Callable[[OrderUpdateEvent | FillEvent], None],
    ) -> None: ...


class PositionProvider(Protocol):
    """按策略逻辑目标键查询有效持仓。"""

    def position(self, strategy_id: str, target_key: str) -> Decimal: ...
