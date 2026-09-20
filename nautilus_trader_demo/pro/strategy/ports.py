"""策略运行框架使用的可插拔端口协议。"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from strategy.contracts import ExecutionRequest


class ExecutionClientPort(Protocol):
    """NT、Bomber、vn.py 和模拟交易客户端共同实现的目标接口。"""

    client_id: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def submit_targets(self, request: ExecutionRequest) -> None: ...

    def cancel_strategy(self, strategy_id: str) -> None: ...


class PositionProvider(Protocol):
    """按策略逻辑目标键查询有效持仓。"""

    def position(self, strategy_id: str, target_key: str) -> Decimal: ...
