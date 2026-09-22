"""P4-CTP1：CTP TraderApi报单字段映射的无网络验证。"""

from __future__ import annotations

from decimal import Decimal

from market.basic.base import InstrumentId
from strategy.execution.contracts import OrderIntent, PositionEffect
from strategy.execution.ctp.native_order import make_ctp_order_insert


def _order(effect: PositionEffect, **changes) -> OrderIntent:
    values = dict(
        strategy_id="alpha", backend_id="ctp-demo",
        instrument_id=InstrumentId.from_str("rb2704.SHFE"), side="SELL",
        quantity=Decimal(2), order_type="LIMIT", price=Decimal(3125),
        position_effect=effect,
    )
    values.update(changes)
    return OrderIntent(**values)


def main() -> None:
    for effect, offset in (
        (PositionEffect.OPEN, "0"), (PositionEffect.CLOSE, "1"),
        (PositionEffect.CLOSE_TODAY, "3"),
        (PositionEffect.CLOSE_YESTERDAY, "4"),
    ):
        payload = make_ctp_order_insert(
            _order(effect), broker_id="broker", investor_id="investor", order_ref="42",
        )
        assert payload["CombOffsetFlag"] == offset
        assert payload["InstrumentID"] == "rb2704" and payload["ExchangeID"] == "SHFE"
        assert payload["Direction"] == "1" and payload["VolumeTotalOriginal"] == 2
    print("P4-CTP1a通过：限价报单保留CTP开仓、平仓、平今和平昨字段")

    invalid = (
        _order(PositionEffect.AUTO),
        _order(PositionEffect.OPEN, reduce_only=True),
        _order(PositionEffect.CLOSE_TODAY, order_type="MARKET", price=None),
        _order(PositionEffect.CLOSE_TODAY, quantity=Decimal("1.5")),
    )
    for order in invalid:
        try:
            make_ctp_order_insert(
                order, broker_id="broker", investor_id="investor", order_ref="42",
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"不能静默接受不安全CTP报单: {order}")
    print("P4-CTP1b通过：AUTO、不受保护市价、reduce-only开仓及非整数手数均拒绝")


if __name__ == "__main__":
    main()
