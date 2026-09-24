"""与策略和具体柜台解耦的统一下单前风险检查。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import threading
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from market.basic.base import Bar, CustomBar, InstrumentId, QuoteTick, TradeTick
from trader.execution.contracts import OrderIntent, OrderSide, PositionEffect
from trader.portfolio import PositionManager


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class KillSwitchMode(str, Enum):
    NORMAL = "NORMAL"
    REDUCE_ONLY = "REDUCE_ONLY"
    HALTED = "HALTED"


class RiskViolationCode(str, Enum):
    KILL_SWITCH = "KILL_SWITCH"
    REDUCE_ONLY = "REDUCE_ONLY"
    ORDER_QUANTITY = "ORDER_QUANTITY"
    POSITION_LIMIT = "POSITION_LIMIT"
    MARKET_PRICE_MISSING = "MARKET_PRICE_MISSING"
    MARKET_STALE = "MARKET_STALE"
    ORDER_NOTIONAL = "ORDER_NOTIONAL"
    POSITION_NOTIONAL = "POSITION_NOTIONAL"


@dataclass(frozen=True)
class RiskViolation:
    code: RiskViolationCode
    instrument_id: InstrumentId
    message: str


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    violations: tuple[RiskViolation, ...] = ()


class RiskRejected(RuntimeError):
    def __init__(self, violations: Sequence[RiskViolation]) -> None:
        self.violations = tuple(violations)
        super().__init__("; ".join(item.message for item in self.violations))


@dataclass(frozen=True)
class RiskLimits:
    """单个标的的前置风险阈值；None表示不启用对应限制。"""

    max_order_quantity: Decimal | int | float | str | None = None
    max_abs_position: Decimal | int | float | str | None = None
    max_order_notional: Decimal | int | float | str | None = None
    max_abs_position_notional: Decimal | int | float | str | None = None
    max_market_age_ns: int | None = None
    contract_multiplier: Decimal | int | float | str = Decimal(1)

    def __post_init__(self) -> None:
        for name in (
            "max_order_quantity",
            "max_abs_position",
            "max_order_notional",
            "max_abs_position_notional",
        ):
            value = getattr(self, name)
            if value is not None:
                normalized = _decimal(value)
                if normalized <= 0:
                    raise ValueError(f"{name}必须大于零")
                object.__setattr__(self, name, normalized)
        multiplier = _decimal(self.contract_multiplier)
        if multiplier <= 0:
            raise ValueError("contract_multiplier必须大于零")
        if self.max_market_age_ns is not None and self.max_market_age_ns < 0:
            raise ValueError("max_market_age_ns不能为负数")
        object.__setattr__(self, "contract_multiplier", multiplier)


@dataclass(frozen=True)
class ReferencePrice:
    price: Decimal
    ts_event: int


class MarketReferencePriceStore:
    """从标准行情事件维护风控使用的最新参考价和事件时间。"""

    def __init__(self) -> None:
        self._prices: dict[InstrumentId, ReferencePrice] = {}
        self._lock = threading.RLock()

    def update(
        self,
        instrument_id: InstrumentId,
        price: Decimal | int | float | str,
        ts_event: int,
    ) -> None:
        normalized = _decimal(price)
        if normalized <= 0:
            raise ValueError("参考价必须大于零")
        if ts_event < 0:
            raise ValueError("ts_event不能为负数")
        with self._lock:
            previous = self._prices.get(instrument_id)
            if previous is not None and ts_event < previous.ts_event:
                return
            self._prices[instrument_id] = ReferencePrice(normalized, ts_event)

    def get(self, instrument_id: InstrumentId) -> ReferencePrice | None:
        with self._lock:
            return self._prices.get(instrument_id)

    def snapshot(self) -> Mapping[InstrumentId, ReferencePrice]:
        with self._lock:
            return MappingProxyType(dict(self._prices))

    def on_market_event(self, event: Any) -> None:
        """供UnifiedStrategyRunner注册；只消费统一标准行情对象。"""
        if isinstance(event, QuoteTick):
            price = (_decimal(event.bid_price) + _decimal(event.ask_price)) / 2
            instrument_id = event.instrument_id
        elif isinstance(event, TradeTick):
            price = _decimal(event.price)
            instrument_id = event.instrument_id
        else:
            base_bar = event.bar if isinstance(event, CustomBar) else event
            if not isinstance(base_bar, Bar):
                raise TypeError(f"不支持的风控行情事件: {type(event).__name__}")
            price = _decimal(base_bar.close)
            instrument_id = base_bar.bar_type.instrument_id
        self.update(instrument_id, price, event.ts_event)


class PreTradeRiskManager:
    """在订单进入Backend前对整个订单批次做原子风险判定。"""

    def __init__(
        self,
        client_id: str,
        position_manager: PositionManager,
        price_store: MarketReferencePriceStore,
        *,
        default_limits: RiskLimits | None = None,
        instrument_limits: Mapping[InstrumentId, RiskLimits] | None = None,
    ) -> None:
        if not client_id.strip():
            raise ValueError("client_id不能为空")
        self.client_id = client_id
        self.position_manager = position_manager
        self.price_store = price_store
        self.default_limits = default_limits or RiskLimits()
        self.instrument_limits = dict(instrument_limits or {})
        self._mode = KillSwitchMode.NORMAL
        self._lock = threading.RLock()

    @property
    def mode(self) -> KillSwitchMode:
        with self._lock:
            return self._mode

    def set_mode(self, mode: KillSwitchMode | str) -> None:
        with self._lock:
            self._mode = KillSwitchMode(mode)

    def evaluate(
        self,
        orders: Sequence[OrderIntent],
        *,
        now_ns: int,
        mode_override: KillSwitchMode | str | None = None,
    ) -> RiskDecision:
        if now_ns < 0:
            raise ValueError("now_ns不能为负数")
        with self._lock:
            mode = _stricter_mode(self._mode, mode_override)
            simulated: dict[InstrumentId, Decimal] = {}
            violations: list[RiskViolation] = []
            for order in orders:
                if order.backend_id != self.client_id:
                    raise ValueError("订单Backend与风控client_id不一致")
                current = simulated.get(
                    order.instrument_id,
                    self.position_manager.effective_position(
                        self.client_id,
                        order.instrument_id,
                    ),
                )
                signed = order.quantity if order.side is OrderSide.BUY else -order.quantity
                projected = current + signed
                limits = self.instrument_limits.get(
                    order.instrument_id,
                    self.default_limits,
                )
                violations.extend(
                    self._evaluate_order(order, current, projected, limits, mode, now_ns),
                )
                simulated[order.instrument_id] = projected
            return RiskDecision(not violations, tuple(violations))

    def check(
        self,
        orders: Sequence[OrderIntent],
        *,
        now_ns: int,
        mode_override: KillSwitchMode | str | None = None,
    ) -> None:
        decision = self.evaluate(
            orders,
            now_ns=now_ns,
            mode_override=mode_override,
        )
        if not decision.allowed:
            raise RiskRejected(decision.violations)

    def _evaluate_order(
        self,
        order: OrderIntent,
        current: Decimal,
        projected: Decimal,
        limits: RiskLimits,
        mode: KillSwitchMode,
        now_ns: int,
    ) -> list[RiskViolation]:
        violations: list[RiskViolation] = []
        reducing = self._is_strictly_reducing(order, current, projected)
        if mode is KillSwitchMode.HALTED:
            violations.append(self._violation(order, RiskViolationCode.KILL_SWITCH, "Kill Switch已停止全部新订单"))
        elif mode is KillSwitchMode.REDUCE_ONLY and not reducing:
            violations.append(self._violation(order, RiskViolationCode.REDUCE_ONLY, "当前仅允许降低绝对仓位"))
        if (order.reduce_only or order.position_effect in {
            PositionEffect.CLOSE,
            PositionEffect.CLOSE_TODAY,
            PositionEffect.CLOSE_YESTERDAY,
        }) and not reducing:
            violations.append(self._violation(order, RiskViolationCode.REDUCE_ONLY, "平仓订单会穿越零轴或增加风险"))
        if limits.max_order_quantity is not None and order.quantity > limits.max_order_quantity:
            violations.append(self._violation(order, RiskViolationCode.ORDER_QUANTITY, f"订单数量{order.quantity}超过上限{limits.max_order_quantity}"))
        if limits.max_abs_position is not None and abs(projected) > limits.max_abs_position:
            violations.append(self._violation(order, RiskViolationCode.POSITION_LIMIT, f"预计仓位{projected}超过绝对上限{limits.max_abs_position}"))

        needs_price = any((
            limits.max_market_age_ns is not None,
            limits.max_order_notional is not None,
            limits.max_abs_position_notional is not None,
        ))
        reference = self.price_store.get(order.instrument_id)
        if needs_price and reference is None:
            violations.append(self._violation(order, RiskViolationCode.MARKET_PRICE_MISSING, "缺少风控参考价"))
            return violations
        if reference is None:
            return violations
        if limits.max_market_age_ns is not None:
            age = now_ns - reference.ts_event
            if age < 0 or age > limits.max_market_age_ns:
                violations.append(self._violation(order, RiskViolationCode.MARKET_STALE, f"行情时间差{age}ns超出允许范围{limits.max_market_age_ns}ns"))
        valuation_price = order.price if order.price is not None else reference.price
        order_notional = order.quantity * valuation_price * limits.contract_multiplier
        position_notional = abs(projected) * reference.price * limits.contract_multiplier
        if limits.max_order_notional is not None and order_notional > limits.max_order_notional:
            violations.append(self._violation(order, RiskViolationCode.ORDER_NOTIONAL, f"订单名义金额{order_notional}超过上限{limits.max_order_notional}"))
        if limits.max_abs_position_notional is not None and position_notional > limits.max_abs_position_notional:
            violations.append(self._violation(order, RiskViolationCode.POSITION_NOTIONAL, f"预计持仓名义金额{position_notional}超过上限{limits.max_abs_position_notional}"))
        return violations

    @staticmethod
    def _is_strictly_reducing(
        order: OrderIntent,
        current: Decimal,
        projected: Decimal,
    ) -> bool:
        if current == 0:
            return False
        return abs(projected) < abs(current) and current * projected >= 0

    @staticmethod
    def _violation(
        order: OrderIntent,
        code: RiskViolationCode,
        message: str,
    ) -> RiskViolation:
        return RiskViolation(code, order.instrument_id, message)


def _stricter_mode(
    configured: KillSwitchMode,
    override: KillSwitchMode | str | None,
) -> KillSwitchMode:
    if override is None:
        return configured
    requested = KillSwitchMode(override)
    severity = {
        KillSwitchMode.NORMAL: 0,
        KillSwitchMode.REDUCE_ONLY: 1,
        KillSwitchMode.HALTED: 2,
    }
    return configured if severity[configured] >= severity[requested] else requested
