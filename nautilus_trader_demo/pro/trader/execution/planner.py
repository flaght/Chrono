"""从账户目标与有效仓位生成中立订单意图。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from trader.contracts import ExecutionRequest
from trader.execution.contracts import OrderIntent, OrderSide, PositionEffect
from trader.portfolio import PositionManager


@dataclass(frozen=True)
class NetTargetOrderPlanner:
    """通用净持仓目标规划器。

    跨越零轴时拆成“先平、后开”两张订单，避免一张reduce-only订单既平仓又反向
    开仓。CTP今昨仓需要更细粒度的Planner，不能使用本类替代。
    """

    position_manager: PositionManager
    planner_id: str = "net-target"

    def plan(self, request: ExecutionRequest) -> Sequence[OrderIntent]:
        orders: list[OrderIntent] = []
        for instrument_id, target in request.targets.items():
            current = self.position_manager.effective_position(
                request.client_id,
                instrument_id,
            )
            target = Decimal(target)
            if target == current:
                continue
            metadata = {
                **dict(request.metadata),
                "target": str(target),
                "current": str(current),
                "request_revision": request.revision,
            }
            if current > 0 and target < current:
                close_quantity = min(current, current - target)
                orders.append(
                    self._order(
                        request,
                        instrument_id,
                        OrderSide.SELL,
                        close_quantity,
                        PositionEffect.CLOSE,
                        True,
                        metadata,
                    ),
                )
                remainder = target - (current - close_quantity)
                if remainder < 0:
                    orders.append(
                        self._order(
                            request,
                            instrument_id,
                            OrderSide.SELL,
                            abs(remainder),
                            PositionEffect.OPEN,
                            False,
                            metadata,
                        ),
                    )
                continue
            if current < 0 and target > current:
                close_quantity = min(abs(current), target - current)
                orders.append(
                    self._order(
                        request,
                        instrument_id,
                        OrderSide.BUY,
                        close_quantity,
                        PositionEffect.CLOSE,
                        True,
                        metadata,
                    ),
                )
                remainder = target - (current + close_quantity)
                if remainder > 0:
                    orders.append(
                        self._order(
                            request,
                            instrument_id,
                            OrderSide.BUY,
                            remainder,
                            PositionEffect.OPEN,
                            False,
                            metadata,
                        ),
                    )
                continue

            delta = target - current
            orders.append(
                self._order(
                    request,
                    instrument_id,
                    OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    abs(delta),
                    PositionEffect.OPEN,
                    False,
                    metadata,
                ),
            )
        return tuple(orders)

    @staticmethod
    def _order(
        request: ExecutionRequest,
        instrument_id,
        side: OrderSide,
        quantity: Decimal,
        effect: PositionEffect,
        reduce_only: bool,
        metadata: dict,
    ) -> OrderIntent:
        return OrderIntent(
            strategy_id=request.strategy_id,
            backend_id=request.client_id,
            instrument_id=instrument_id,
            side=side,
            quantity=quantity,
            position_effect=effect,
            reduce_only=reduce_only,
            metadata=metadata,
        )
