"""执行后端、订单规划和成交回报使用的中立数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from market.basic.base import InstrumentId


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class ExecutionBackendKind(str, Enum):
    """执行后端运行性质。"""

    SIMULATION = "simulation"
    LIVE = "live"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PositionEffect(str, Enum):
    """开平语义；AUTO 由具体市场的 OrderPlanner 决定。"""

    AUTO = "AUTO"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    CLOSE_TODAY = "CLOSE_TODAY"
    CLOSE_YESTERDAY = "CLOSE_YESTERDAY"


class ExecutionReportType(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"


class AmbiguousOrderSubmission(RuntimeError):
    """柜台回报已到达，但发送调用仍失败；订单状态须以柜台查询为准。"""

    order_may_be_live = True


@dataclass(frozen=True)
class AccountPositionSnapshot:
    """交易柜台返回的账户级权威净仓快照。"""

    backend_id: str
    revision: int
    ts_event: int
    positions: Mapping[InstrumentId, Decimal | int | float | str]

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id不能为空")
        if self.revision < 1:
            raise ValueError("revision必须为正整数")
        if self.ts_event < 0:
            raise ValueError("ts_event不能为负数")
        normalized = {
            instrument_id: _decimal(quantity)
            for instrument_id, quantity in self.positions.items()
        }
        object.__setattr__(self, "positions", MappingProxyType(normalized))


@dataclass(frozen=True)
class OrderIntent:
    """OrderPlanner 输出、ExecutionBackend 消费的中立订单意图。"""

    strategy_id: str
    backend_id: str
    instrument_id: InstrumentId
    side: OrderSide | str
    quantity: Decimal | int | float | str
    order_type: OrderType | str = OrderType.MARKET
    price: Decimal | int | float | str | None = None
    position_effect: PositionEffect | str = PositionEffect.AUTO
    reduce_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy_id.strip() or not self.backend_id.strip():
            raise ValueError("strategy_id 和 backend_id 不能为空")
        side = OrderSide(self.side)
        order_type = OrderType(self.order_type)
        effect = PositionEffect(self.position_effect)
        quantity = _decimal(self.quantity)
        price = None if self.price is None else _decimal(self.price)
        if quantity <= 0:
            raise ValueError("订单数量必须大于零")
        if order_type is OrderType.LIMIT and (price is None or price <= 0):
            raise ValueError("限价单必须提供大于零的价格")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "position_effect", effect)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ExecutionReport:
    """模拟撮合器和真实交易客户端共同返回的标准执行回报。"""

    backend_id: str
    client_order_id: str
    instrument_id: InstrumentId
    report_type: ExecutionReportType | str
    ts_event: int
    filled_quantity: Decimal | int | float | str = Decimal(0)
    fill_price: Decimal | int | float | str | None = None
    order_side: OrderSide | str | None = None
    order_quantity: Decimal | int | float | str | None = None
    position_effect: PositionEffect | str | None = None
    reason: str | None = None
    report_id: str | None = None
    sequence: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend_id.strip() or not self.client_order_id.strip():
            raise ValueError("backend_id 和 client_order_id 不能为空")
        if self.ts_event < 0:
            raise ValueError("ts_event 不能为负数")
        report_type = ExecutionReportType(self.report_type)
        filled = _decimal(self.filled_quantity)
        fill_price = None if self.fill_price is None else _decimal(self.fill_price)
        order_side = None if self.order_side is None else OrderSide(self.order_side)
        order_quantity = (
            None if self.order_quantity is None else _decimal(self.order_quantity)
        )
        position_effect = (
            None
            if self.position_effect is None
            else PositionEffect(self.position_effect)
        )
        if filled < 0:
            raise ValueError("filled_quantity 不能为负数")
        if fill_price is not None and fill_price <= 0:
            raise ValueError("fill_price 必须大于零")
        if order_quantity is not None and order_quantity <= 0:
            raise ValueError("order_quantity 必须大于零")
        if self.report_id is not None and not self.report_id.strip():
            raise ValueError("report_id不能为空字符串")
        if self.sequence is not None and self.sequence < 1:
            raise ValueError("sequence必须为正整数")
        object.__setattr__(self, "report_type", report_type)
        object.__setattr__(self, "filled_quantity", filled)
        object.__setattr__(self, "fill_price", fill_price)
        object.__setattr__(self, "order_side", order_side)
        object.__setattr__(self, "order_quantity", order_quantity)
        object.__setattr__(self, "position_effect", position_effect)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
