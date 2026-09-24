"""CTP双向持仓、今昨仓与逐日盯市账本。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from market.basic.base import InstrumentId
from trader.execution.contracts import OrderSide, PositionEffect


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass
class _Bucket:
    quantity: Decimal = Decimal(0)
    basis_price: Decimal = Decimal(0)

    def add(self, quantity: Decimal, price: Decimal) -> None:
        total = self.quantity + quantity
        if total <= 0:
            raise ValueError("持仓数量必须大于零")
        self.basis_price = (
            self.quantity * self.basis_price + quantity * price
        ) / total
        self.quantity = total

    def remove(self, quantity: Decimal) -> None:
        if quantity <= 0 or quantity > self.quantity:
            raise ValueError("平仓数量超过对应今昨仓")
        self.quantity -= quantity
        if self.quantity == 0:
            self.basis_price = Decimal(0)


@dataclass
class _Position:
    long_today: _Bucket
    long_yesterday: _Bucket
    short_today: _Bucket
    short_yesterday: _Bucket

    @classmethod
    def empty(cls) -> _Position:
        return cls(_Bucket(), _Bucket(), _Bucket(), _Bucket())


@dataclass(frozen=True)
class CtpPositionSnapshot:
    instrument_id: InstrumentId
    long_today: Decimal
    long_yesterday: Decimal
    short_today: Decimal
    short_yesterday: Decimal
    long_today_basis: Decimal
    long_yesterday_basis: Decimal
    short_today_basis: Decimal
    short_yesterday_basis: Decimal

    @property
    def long_total(self) -> Decimal:
        return self.long_today + self.long_yesterday

    @property
    def short_total(self) -> Decimal:
        return self.short_today + self.short_yesterday

    @property
    def net_position(self) -> Decimal:
        return self.long_total - self.short_total


@dataclass(frozen=True)
class CtpLedgerState:
    trading_day: str | None
    positions: Mapping[InstrumentId, CtpPositionSnapshot]

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))


@dataclass(frozen=True)
class CtpCloseAllocation:
    position_effect: PositionEffect
    quantity: Decimal
    basis_price: Decimal


@dataclass(frozen=True)
class CtpFillResult:
    realized_pnl: Decimal
    allocations: tuple[CtpCloseAllocation, ...] = ()


@dataclass(frozen=True)
class CtpSettlementResult:
    trading_day: str
    variation_margin: Decimal
    by_instrument: Mapping[InstrumentId, Decimal]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_instrument", MappingProxyType(dict(self.by_instrument)))


class CtpPositionLedger:
    """记录CTP账户双向持仓，并严格区分今仓和昨仓。"""

    def __init__(self, trading_day: str | None = None) -> None:
        self._trading_day = trading_day
        self._positions: dict[InstrumentId, _Position] = {}
        self._lock = threading.RLock()

    @property
    def trading_day(self) -> str | None:
        return self._trading_day

    def snapshot(self, instrument_id: InstrumentId) -> CtpPositionSnapshot:
        with self._lock:
            position = self._positions.get(instrument_id, _Position.empty())
            return CtpPositionSnapshot(
                instrument_id=instrument_id,
                long_today=position.long_today.quantity,
                long_yesterday=position.long_yesterday.quantity,
                short_today=position.short_today.quantity,
                short_yesterday=position.short_yesterday.quantity,
                long_today_basis=position.long_today.basis_price,
                long_yesterday_basis=position.long_yesterday.basis_price,
                short_today_basis=position.short_today.basis_price,
                short_yesterday_basis=position.short_yesterday.basis_price,
            )

    def state(self) -> CtpLedgerState:
        with self._lock:
            return CtpLedgerState(
                trading_day=self._trading_day,
                positions={
                    instrument_id: self.snapshot(instrument_id)
                    for instrument_id in self._positions
                },
            )

    def restore(self, state: CtpLedgerState) -> None:
        if state.trading_day is not None and not state.trading_day.strip():
            raise ValueError("trading_day不能为空字符串")
        restored: dict[InstrumentId, _Position] = {}
        for instrument_id, snapshot in state.positions.items():
            if snapshot.instrument_id != instrument_id:
                raise ValueError("CTP仓位快照键与instrument_id不一致")
            values = (
                snapshot.long_today,
                snapshot.long_yesterday,
                snapshot.short_today,
                snapshot.short_yesterday,
            )
            bases = (
                snapshot.long_today_basis,
                snapshot.long_yesterday_basis,
                snapshot.short_today_basis,
                snapshot.short_yesterday_basis,
            )
            if any(value < 0 for value in values):
                raise ValueError("CTP恢复仓位不能为负数")
            for quantity, basis in zip(values, bases):
                if quantity > 0 and basis <= 0:
                    raise ValueError("非零CTP恢复仓位必须具有正成本价")
                if quantity == 0 and basis != 0:
                    raise ValueError("零CTP恢复仓位的成本价必须为零")
            restored[instrument_id] = _Position(
                _Bucket(snapshot.long_today, snapshot.long_today_basis),
                _Bucket(snapshot.long_yesterday, snapshot.long_yesterday_basis),
                _Bucket(snapshot.short_today, snapshot.short_today_basis),
                _Bucket(snapshot.short_yesterday, snapshot.short_yesterday_basis),
            )
        with self._lock:
            self._trading_day = state.trading_day
            self._positions = restored

    def apply_fill(
        self,
        instrument_id: InstrumentId,
        side: OrderSide | str,
        position_effect: PositionEffect | str,
        quantity: Decimal | int | float | str,
        price: Decimal | int | float | str,
        multiplier: Decimal | int | float | str,
        *,
        close_today_first: bool = False,
    ) -> CtpFillResult:
        side = OrderSide(side)
        effect = PositionEffect(position_effect)
        quantity = _decimal(quantity)
        price = _decimal(price)
        multiplier = _decimal(multiplier)
        if quantity <= 0 or price <= 0 or multiplier <= 0:
            raise ValueError("quantity、price和multiplier必须大于零")
        if effect is PositionEffect.AUTO:
            raise ValueError("CTP成交必须携带明确OPEN/CLOSE/CLOSE_TODAY/CLOSE_YESTERDAY")

        with self._lock:
            position = self._positions.setdefault(instrument_id, _Position.empty())
            if effect is PositionEffect.OPEN:
                bucket = position.long_today if side is OrderSide.BUY else position.short_today
                bucket.add(quantity, price)
                return CtpFillResult(Decimal(0))

            today = position.short_today if side is OrderSide.BUY else position.long_today
            yesterday = (
                position.short_yesterday if side is OrderSide.BUY else position.long_yesterday
            )
            if effect is PositionEffect.CLOSE_TODAY:
                candidates = ((PositionEffect.CLOSE_TODAY, today),)
            elif effect is PositionEffect.CLOSE_YESTERDAY:
                candidates = ((PositionEffect.CLOSE_YESTERDAY, yesterday),)
            elif close_today_first:
                candidates = (
                    (PositionEffect.CLOSE_TODAY, today),
                    (PositionEffect.CLOSE_YESTERDAY, yesterday),
                )
            else:
                candidates = (
                    (PositionEffect.CLOSE_YESTERDAY, yesterday),
                    (PositionEffect.CLOSE_TODAY, today),
                )

            if sum(bucket.quantity for _, bucket in candidates) < quantity:
                raise ValueError("平仓数量超过可用反向持仓")
            remaining = quantity
            realized = Decimal(0)
            allocations: list[CtpCloseAllocation] = []
            for allocated_effect, bucket in candidates:
                allocated = min(remaining, bucket.quantity)
                if allocated == 0:
                    continue
                basis = bucket.basis_price
                unit_pnl = price - basis if side is OrderSide.SELL else basis - price
                realized += unit_pnl * allocated * multiplier
                bucket.remove(allocated)
                allocations.append(CtpCloseAllocation(allocated_effect, allocated, basis))
                remaining -= allocated
                if remaining == 0:
                    break
            return CtpFillResult(realized, tuple(allocations))

    def settle(
        self,
        trading_day: str,
        settlement_prices: Mapping[InstrumentId, Decimal | int | float | str],
        multipliers: Mapping[InstrumentId, Decimal | int | float | str],
    ) -> CtpSettlementResult:
        """按结算价逐日盯市，并把所有今仓滚为下一交易日昨仓。"""

        normalized_day = trading_day.strip()
        if not normalized_day:
            raise ValueError("trading_day不能为空")
        with self._lock:
            if self._trading_day is not None and normalized_day <= self._trading_day:
                raise ValueError("新交易日必须晚于当前交易日")
            normalized_inputs: dict[InstrumentId, tuple[Decimal, Decimal]] = {}
            # 先完整校验，再修改任何仓位，保证多合约结算具有原子性。
            for instrument_id in self._positions:
                if instrument_id not in settlement_prices or instrument_id not in multipliers:
                    raise ValueError(f"缺少结算价或合约乘数: {instrument_id}")
                settlement = _decimal(settlement_prices[instrument_id])
                multiplier = _decimal(multipliers[instrument_id])
                if settlement <= 0 or multiplier <= 0:
                    raise ValueError("结算价和合约乘数必须大于零")
                normalized_inputs[instrument_id] = (settlement, multiplier)
            by_instrument: dict[InstrumentId, Decimal] = {}
            for instrument_id, position in self._positions.items():
                settlement, multiplier = normalized_inputs[instrument_id]
                pnl = Decimal(0)
                for bucket in (position.long_today, position.long_yesterday):
                    pnl += (settlement - bucket.basis_price) * bucket.quantity * multiplier
                for bucket in (position.short_today, position.short_yesterday):
                    pnl += (bucket.basis_price - settlement) * bucket.quantity * multiplier
                by_instrument[instrument_id] = pnl
                long_quantity = position.long_today.quantity + position.long_yesterday.quantity
                short_quantity = position.short_today.quantity + position.short_yesterday.quantity
                position.long_yesterday = _Bucket(long_quantity, settlement if long_quantity else Decimal(0))
                position.short_yesterday = _Bucket(short_quantity, settlement if short_quantity else Decimal(0))
                position.long_today = _Bucket()
                position.short_today = _Bucket()
            self._trading_day = normalized_day
            return CtpSettlementResult(
                trading_day=normalized_day,
                variation_margin=sum(by_instrument.values(), Decimal(0)),
                by_instrument=by_instrument,
            )
