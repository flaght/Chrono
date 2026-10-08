"""CTP 今昨仓规划后的保护限价与分阶段反手。"""

from dataclasses import replace
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from bomber.framework.trader.execution.contracts import OrderSide, OrderType, PositionEffect
from bomber.framework.trader.execution.ctp.planner import CtpClosePlanner


class CtpLimitPlanner:
    """先平旧方向，确认成交后由后续目标推进开新方向。"""

    def __init__(self, ledger, positions, prices, tick_size, offset_ticks):
        tick_size = Decimal(str(tick_size))
        if not tick_size.is_finite() or tick_size <= 0:
            raise ValueError("CTP价格步长须为正且有限")
        if isinstance(offset_ticks, bool) or not isinstance(offset_ticks, int) or offset_ticks < 0:
            raise ValueError("CTP限价偏移须为非负整数")
        self._planner = CtpClosePlanner(ledger, close_today_first=True)
        self._positions = positions
        self._prices = prices
        self._tick_size = tick_size
        self._offset_ticks = offset_ticks

    def plan(self, request):
        planned = tuple(self._planner.plan(request))
        closing = {order.instrument_id for order in planned
                   if order.position_effect is not PositionEffect.OPEN}
        result = []
        for order in planned:
            if self._positions.working_quantity(request.client_id, order.instrument_id):
                continue
            if order.position_effect is PositionEffect.OPEN and order.instrument_id in closing:
                continue
            reference = self._prices.get(order.instrument_id)
            if reference is None:
                raise RuntimeError("CTP缺少行情参考价，拒绝报单")
            raw = reference.price + self._tick_size * self._offset_ticks * (
                1 if order.side is OrderSide.BUY else -1)
            rounding = ROUND_CEILING if order.side is OrderSide.BUY else ROUND_FLOOR
            price = (raw / self._tick_size).to_integral_value(rounding=rounding) * self._tick_size
            if not price.is_finite() or price <= 0:
                raise RuntimeError("CTP限价无效")
            result.append(replace(order, order_type=OrderType.LIMIT, price=price))
        return tuple(result)
