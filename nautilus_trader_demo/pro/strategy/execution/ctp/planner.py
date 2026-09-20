"""按CTP今昨仓生成明确开平标志的订单规划器。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from strategy.contracts import ExecutionRequest
from strategy.execution.contracts import OrderIntent, OrderSide, PositionEffect
from strategy.execution.ctp.ledger import CtpPositionLedger


@dataclass(frozen=True)
class CtpClosePlanner:
    ledger: CtpPositionLedger
    close_today_first: bool = False
    planner_id: str = "ctp-close-offset"

    def plan(self, request: ExecutionRequest) -> Sequence[OrderIntent]:
        orders: list[OrderIntent] = []
        for instrument_id, target_value in request.targets.items():
            target = Decimal(target_value)
            snapshot = self.ledger.snapshot(instrument_id)
            delta = target - snapshot.net_position
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            required = abs(delta)
            today_available = (
                snapshot.short_today if side is OrderSide.BUY else snapshot.long_today
            )
            yesterday_available = (
                snapshot.short_yesterday
                if side is OrderSide.BUY
                else snapshot.long_yesterday
            )
            candidates = (
                (
                    PositionEffect.CLOSE_TODAY,
                    today_available,
                ),
                (
                    PositionEffect.CLOSE_YESTERDAY,
                    yesterday_available,
                ),
            )
            if not self.close_today_first:
                candidates = tuple(reversed(candidates))
            metadata = {
                **dict(request.metadata),
                "target": str(target),
                "net_position": str(snapshot.net_position),
                "request_revision": request.revision,
            }
            for effect, available in candidates:
                quantity = min(required, available)
                if quantity <= 0:
                    continue
                orders.append(
                    OrderIntent(
                        strategy_id=request.strategy_id,
                        backend_id=request.client_id,
                        instrument_id=instrument_id,
                        side=side,
                        quantity=quantity,
                        position_effect=effect,
                        reduce_only=True,
                        metadata=metadata,
                    ),
                )
                required -= quantity
            if required > 0:
                orders.append(
                    OrderIntent(
                        strategy_id=request.strategy_id,
                        backend_id=request.client_id,
                        instrument_id=instrument_id,
                        side=side,
                        quantity=required,
                        position_effect=PositionEffect.OPEN,
                        reduce_only=False,
                        metadata=metadata,
                    ),
                )
        return tuple(orders)
