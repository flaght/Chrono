"""模拟与实盘交易端共用的双向执行事件契约。

事件契约本身不导入任何柜台SDK、行情模型或撮合引擎。P1的旧回报转换函数
也集中于此，但只在调用转换函数时加载旧执行类型；状态处理和策略分发仍由客户端负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from trader.execution.contracts import ExecutionReport
    from trader.execution.order_state import OrderReportUpdate


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}不能为空")
    return value.strip()


def _optional(value: str | None, name: str) -> str | None:
    return None if value is None else _required(value, name)


def _decimal(value: Decimal | int | float | str, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name}不是有效数值")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (ValueError, ArithmeticError, TypeError) as exc:
        raise ValueError(f"{name}不是有效数值") from exc
    if not result.is_finite():
        raise ValueError(f"{name}必须为有限数值")
    return result


def _timestamp(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name}必须为非负整数纳秒")
    return value


def _revision(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("revision必须为正整数")
    return value


class OrderEventStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"


class ExecutionSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class FillPositionEffect(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    CLOSE_TODAY = "CLOSE_TODAY"
    CLOSE_YESTERDAY = "CLOSE_YESTERDAY"


@dataclass(frozen=True)
class ExecutionIdentity:
    """统一客户端订单ID是回报归属键；原生委托号可在受理后才出现。"""

    client_id: str
    account_id: str
    strategy_id: str
    client_order_id: str
    instrument_id: str
    venue_order_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("client_id", "account_id", "strategy_id", "client_order_id", "instrument_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        object.__setattr__(self, "venue_order_id", _optional(self.venue_order_id, "venue_order_id"))


@dataclass(frozen=True)
class OrderUpdateEvent:
    """一条订单状态事实；累计成交量，不把它误当成本次增量。"""

    identity: ExecutionIdentity
    event_id: str
    status: OrderEventStatus | str
    side: ExecutionSide | str
    order_quantity: Decimal | int | float | str
    cumulative_filled: Decimal | int | float | str
    ts_event: int
    sequence: int | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise TypeError("identity必须为ExecutionIdentity")
        object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))
        status = OrderEventStatus(self.status)
        side = ExecutionSide(self.side)
        quantity = _decimal(self.order_quantity, "order_quantity")
        filled = _decimal(self.cumulative_filled, "cumulative_filled")
        if quantity <= 0 or filled < 0 or filled > quantity:
            raise ValueError("订单数量与累计成交量不一致")
        if status is OrderEventStatus.FILLED and filled != quantity:
            raise ValueError("FILLED事件的累计成交量必须等于订单数量")
        if status is OrderEventStatus.PARTIALLY_FILLED and not 0 < filled < quantity:
            raise ValueError("PARTIALLY_FILLED事件必须具有部分成交量")
        if status in {OrderEventStatus.SUBMITTED, OrderEventStatus.ACCEPTED} and filled != 0:
            raise ValueError("提交/受理事件不能带累计成交量")
        if self.sequence is not None and (
            not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1
        ):
            raise ValueError("sequence必须为正整数")
        reason = _optional(self.reason, "reason")
        if status is OrderEventStatus.REJECTED and reason is None:
            raise ValueError("拒单事件必须说明原因")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "order_quantity", quantity)
        object.__setattr__(self, "cumulative_filled", filled)
        object.__setattr__(self, "ts_event", _timestamp(self.ts_event, "ts_event"))
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def dedupe_key(self) -> tuple[str, str]:
        return self.identity.client_id, self.event_id


@dataclass(frozen=True)
class FillEvent:
    """一次成交增量；累计量和费用若柜台尚未提供，必须显式保持None。"""

    identity: ExecutionIdentity
    event_id: str
    trade_id: str
    side: ExecutionSide | str
    quantity: Decimal | int | float | str
    price: Decimal | int | float | str
    commission: Decimal | int | float | str | None
    commission_currency: str | None
    ts_event: int
    cumulative_filled: Decimal | int | float | str | None = None
    sequence: int | None = None
    position_effect: FillPositionEffect | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise TypeError("identity必须为ExecutionIdentity")
        object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))
        object.__setattr__(self, "trade_id", _required(self.trade_id, "trade_id"))
        side = ExecutionSide(self.side)
        quantity = _decimal(self.quantity, "quantity")
        price = _decimal(self.price, "price")
        commission = None if self.commission is None else _decimal(self.commission, "commission")
        currency = _optional(self.commission_currency, "commission_currency")
        cumulative = (
            None if self.cumulative_filled is None
            else _decimal(self.cumulative_filled, "cumulative_filled")
        )
        if quantity <= 0 or price <= 0:
            raise ValueError("成交数量/价格必须为正")
        if commission is not None and commission < 0:
            raise ValueError("手续费不能为负")
        if (commission is None) != (currency is None):
            raise ValueError("手续费与币种必须同时已知或同时未知")
        if cumulative is not None and cumulative < quantity:
            raise ValueError("累计成交量不能小于本次成交量")
        if self.sequence is not None and (
            not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1
        ):
            raise ValueError("sequence必须为正整数")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "commission", commission)
        object.__setattr__(self, "commission_currency", currency)
        object.__setattr__(self, "cumulative_filled", cumulative)
        object.__setattr__(self, "ts_event", _timestamp(self.ts_event, "ts_event"))
        object.__setattr__(
            self, "position_effect", None if self.position_effect is None
            else FillPositionEffect(self.position_effect),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def dedupe_key(self) -> tuple[str, str, str, str]:
        # 柜台TradeID可能只在订单/交易日范围内唯一，不能仅凭TradeID跨订单去重。
        return (self.identity.client_id, self.identity.account_id,
                self.identity.client_order_id, self.trade_id)


@dataclass(frozen=True)
class InstrumentPosition:
    """账户级合约仓位；今昨和双向字段未知时必须保持None，不猜测。"""

    net_quantity: Decimal | int | float | str
    long_quantity: Decimal | int | float | str | None = None
    short_quantity: Decimal | int | float | str | None = None
    long_today: Decimal | int | float | str | None = None
    long_yesterday: Decimal | int | float | str | None = None
    short_today: Decimal | int | float | str | None = None
    short_yesterday: Decimal | int | float | str | None = None

    def __post_init__(self) -> None:
        net = _decimal(self.net_quantity, "net_quantity")
        object.__setattr__(self, "net_quantity", net)
        names = ("long_quantity", "short_quantity", "long_today", "long_yesterday",
                 "short_today", "short_yesterday")
        for name in names:
            value = getattr(self, name)
            if value is None:
                continue
            normalized = _decimal(value, name)
            if normalized < 0:
                raise ValueError(f"{name}不能为负")
            object.__setattr__(self, name, normalized)
        if (self.long_quantity is None) != (self.short_quantity is None):
            raise ValueError("long_quantity和short_quantity必须同时提供或同时未知")
        if self.long_quantity is not None and self.long_quantity - self.short_quantity != net:
            raise ValueError("多空仓与净仓不一致")
        for prefix in ("long", "short"):
            total = getattr(self, f"{prefix}_quantity")
            today = getattr(self, f"{prefix}_today")
            yesterday = getattr(self, f"{prefix}_yesterday")
            if (today is None) != (yesterday is None):
                raise ValueError(f"{prefix}今昨仓必须成对提供")
            if today is not None and total is None:
                raise ValueError(f"{prefix}今昨仓需要对应总仓")
            if today is not None and today + yesterday != total:
                raise ValueError(f"{prefix}今昨仓之和与总仓不一致")


@dataclass(frozen=True)
class AccountPositionEvent:
    """账户权威全量仓位快照；空映射表示确认空仓，不是查询失败。"""

    client_id: str
    account_id: str
    revision: int
    ts_event: int
    positions: Mapping[str, InstrumentPosition]

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_id", _required(self.client_id, "client_id"))
        object.__setattr__(self, "account_id", _required(self.account_id, "account_id"))
        object.__setattr__(self, "revision", _revision(self.revision))
        object.__setattr__(self, "ts_event", _timestamp(self.ts_event, "ts_event"))
        normalized = {}
        for instrument_id, position in self.positions.items():
            if not isinstance(position, InstrumentPosition):
                raise TypeError("positions值必须为InstrumentPosition")
            normalized[_required(instrument_id, "instrument_id")] = position
        object.__setattr__(self, "positions", MappingProxyType(normalized))


@dataclass(frozen=True)
class CurrencyBalance:
    """每种币/货币的账户指标；None表示柜台未提供，不补零。"""

    currency: str
    total: Decimal | int | float | str | None = None
    equity: Decimal | int | float | str | None = None
    available: Decimal | int | float | str | None = None
    margin_used: Decimal | int | float | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", _required(self.currency, "currency"))
        if all(getattr(self, name) is None for name in ("total", "equity", "available", "margin_used")):
            raise ValueError("账户余额至少需要一个已知指标")
        for name in ("total", "equity", "available", "margin_used"):
            value = getattr(self, name)
            if value is None:
                continue
            normalized = _decimal(value, name)
            if name == "margin_used" and normalized < 0:
                raise ValueError("margin_used不能为负")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class AccountStateEvent:
    """账户级资金/保证金全量快照，不自动归属到某个策略。"""

    client_id: str
    account_id: str
    revision: int
    ts_event: int
    balances: Mapping[str, CurrencyBalance]

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_id", _required(self.client_id, "client_id"))
        object.__setattr__(self, "account_id", _required(self.account_id, "account_id"))
        object.__setattr__(self, "revision", _revision(self.revision))
        object.__setattr__(self, "ts_event", _timestamp(self.ts_event, "ts_event"))
        normalized = {}
        for currency, balance in self.balances.items():
            key = _required(currency, "currency")
            if not isinstance(balance, CurrencyBalance) or key != balance.currency:
                raise ValueError("balances键与CurrencyBalance.currency必须一致")
            normalized[key] = balance
        if not normalized:
            raise ValueError("账户快照不能没有任何货币余额")
        object.__setattr__(self, "balances", MappingProxyType(normalized))


@dataclass(frozen=True)
class ActiveOrder:
    """柜台仍在工作的订单；数量为绝对值，方向决定在途量符号。"""

    client_order_id: str
    instrument_id: str
    side: ExecutionSide | str
    order_quantity: Decimal | int | float | str
    cumulative_filled: Decimal | int | float | str
    remaining_quantity: Decimal | int | float | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_order_id", _required(self.client_order_id, "client_order_id"))
        object.__setattr__(self, "instrument_id", _required(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "side", ExecutionSide(self.side))
        quantity = _decimal(self.order_quantity, "order_quantity")
        filled = _decimal(self.cumulative_filled, "cumulative_filled")
        remaining = _decimal(self.remaining_quantity, "remaining_quantity")
        if quantity <= 0 or filled < 0 or remaining <= 0 or filled + remaining != quantity:
            raise ValueError("活动订单数量不守恒或无剩余量")
        object.__setattr__(self, "order_quantity", quantity)
        object.__setattr__(self, "cumulative_filled", filled)
        object.__setattr__(self, "remaining_quantity", remaining)


@dataclass(frozen=True)
class ActiveOrderSnapshot:
    """柜台权威全量活动订单快照；空列表须表示查询确认无活动订单。"""

    client_id: str
    account_id: str
    revision: int
    ts_event: int
    orders: tuple[ActiveOrder, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_id", _required(self.client_id, "client_id"))
        object.__setattr__(self, "account_id", _required(self.account_id, "account_id"))
        object.__setattr__(self, "revision", _revision(self.revision))
        object.__setattr__(self, "ts_event", _timestamp(self.ts_event, "ts_event"))
        orders = tuple(self.orders)
        if any(not isinstance(order, ActiveOrder) for order in orders):
            raise TypeError("orders必须为ActiveOrder序列")
        if len({order.client_order_id for order in orders}) != len(orders):
            raise ValueError("活动订单快照包含重复client_order_id")
        object.__setattr__(self, "orders", orders)


def map_applied_report(
    report: ExecutionReport,
    update: OrderReportUpdate,
    *,
    account_id: str,
) -> tuple[OrderUpdateEvent, FillEvent | None]:
    """把状态机已接受的旧回报转换为订单事件及可选成交事件。

    缺少策略归属或成交ID时拒绝猜测。旧类型仅在此函数被调用时加载，
    不影响事件契约独立用于纯内存验证。
    """
    from trader.execution.contracts import ExecutionReportType, PositionEffect

    if not update.applied:
        raise ValueError("未应用的回报不能生成策略事件")
    strategy_id = report.metadata.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise ValueError("回报缺少可信的strategy_id关联")
    reported_account = report.metadata.get("account_id")
    if reported_account is not None and reported_account != account_id:
        raise ValueError("回报account_id与执行客户端账户不一致")
    if report.report_id is None and report.sequence is None:
        raise ValueError("回报缺少稳定的事件ID/序号")
    event_id = report.report_id or f"{report.client_order_id}:{report.sequence}"
    identity = ExecutionIdentity(
        client_id=report.backend_id,
        account_id=account_id,
        strategy_id=strategy_id,
        client_order_id=report.client_order_id,
        instrument_id=str(report.instrument_id),
        venue_order_id=report.metadata.get("venue_order_id"),
    )
    status = {
        ExecutionReportType.ACCEPTED: OrderEventStatus.ACCEPTED,
        ExecutionReportType.PARTIALLY_FILLED: OrderEventStatus.PARTIALLY_FILLED,
        ExecutionReportType.FILLED: OrderEventStatus.FILLED,
        ExecutionReportType.REJECTED: OrderEventStatus.REJECTED,
        ExecutionReportType.CANCELED: OrderEventStatus.CANCELED,
    }[report.report_type]
    order = OrderUpdateEvent(
        identity=identity,
        event_id=event_id,
        status=status,
        side=update.state.side.value,
        order_quantity=update.state.order_quantity,
        cumulative_filled=update.state.filled_quantity,
        ts_event=report.ts_event,
        sequence=report.sequence,
        reason=report.reason,
        metadata=report.metadata,
    )
    if not update.fill_delta:
        return order, None
    trade_id = report.metadata.get("trade_id")
    if not isinstance(trade_id, str) or not trade_id.strip():
        raise ValueError("成交回报缺少trade_id，不可凭价格/时间伪造")
    if report.fill_price is None:
        raise ValueError("成交回报缺少价格")
    raw_commission = report.metadata.get("commission")
    currency = report.metadata.get("commission_currency")
    if raw_commission is not None and currency is None:
        parts = str(raw_commission).split()
        if len(parts) == 2:
            raw_commission, currency = parts
        else:
            raise ValueError("手续费存在但没有明确币种")
    effect = report.position_effect
    if effect is PositionEffect.AUTO:
        effect = None
    fill = FillEvent(
        identity=identity,
        event_id=f"{event_id}:fill",
        trade_id=trade_id,
        side=update.state.side.value,
        quantity=update.fill_delta,
        price=report.fill_price,
        commission=None if raw_commission is None else Decimal(str(raw_commission)),
        commission_currency=currency,
        ts_event=report.ts_event,
        cumulative_filled=update.state.filled_quantity,
        sequence=report.sequence,
        position_effect=None if effect is None else effect.value,
        metadata=report.metadata,
    )
    return order, fill
