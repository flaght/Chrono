"""中立订单到CTP ReqOrderInsert的严格字段转换；不连接柜台。"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from strategy.execution.contracts import OrderIntent, OrderSide, OrderType, PositionEffect


_OFFSET = {
    PositionEffect.OPEN: "0",             # THOST_FTDC_OF_Open
    PositionEffect.CLOSE: "1",            # THOST_FTDC_OF_Close
    PositionEffect.CLOSE_TODAY: "3",      # THOST_FTDC_OF_CloseToday
    PositionEffect.CLOSE_YESTERDAY: "4",  # THOST_FTDC_OF_CloseYesterday
}


def make_ctp_order_insert(
    order: OrderIntent,
    *,
    broker_id: str,
    investor_id: str,
    order_ref: str,
) -> dict[str, Any]:
    """只生成显式限价/GFD订单；报单与回报关联由Trader Driver负责。

    不猜测AUTO开平、不把市价单静默改成不受保护的限价单，也不把
    reduceOnly当作CTP原生布尔字段。平今/平昨必须先经CTP Planner决策。
    """
    if not broker_id.strip() or not investor_id.strip() or not order_ref.strip():
        raise ValueError("CTP报单必须指定BrokerID、InvestorID和OrderRef")
    if order.order_type is not OrderType.LIMIT or order.price is None:
        raise ValueError("原生CTP首阶段只支持显式限价单")
    if not order.price.is_finite() or order.price <= 0 or not order.quantity.is_finite():
        raise ValueError("CTP价格和手数必须为有限正数")
    if order.position_effect not in _OFFSET:
        raise ValueError("CTP订单必须指定明确开仓/平仓/平今/平昨")
    if order.reduce_only and order.position_effect is PositionEffect.OPEN:
        raise ValueError("reduce_only订单不能映射为CTP开仓")
    if order.quantity != order.quantity.to_integral_value() or order.quantity > 2_147_483_647:
        raise ValueError("CTP报单手数必须为有效正整数")
    symbol, separator, venue = str(order.instrument_id).rpartition(".")
    if not separator or not symbol or not venue:
        raise ValueError("CTP合约ID必须包含明确交易所后缀")
    return {
        "BrokerID": broker_id,
        "InvestorID": investor_id,
        "UserID": investor_id,
        "InstrumentID": symbol,
        "ExchangeID": venue,
        "OrderRef": order_ref,
        "Direction": "0" if order.side is OrderSide.BUY else "1",
        "CombOffsetFlag": _OFFSET[order.position_effect],
        "CombHedgeFlag": "1",  # 首阶段仅投机；套保/套利需独立能力配置
        "OrderPriceType": "2",  # 限价
        "LimitPrice": float(Decimal(order.price)),
        "VolumeTotalOriginal": int(order.quantity),
        "TimeCondition": "3",  # GFD
        "VolumeCondition": "1",  # 任意成交量
        "MinVolume": 1,
        "ContingentCondition": "1",  # 立即
        "ForceCloseReason": "0",  # 非强平
        "IsAutoSuspend": 0,
        "UserForceClose": 0,
    }
