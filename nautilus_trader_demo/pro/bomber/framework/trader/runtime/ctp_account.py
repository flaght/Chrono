"""CTP权威今昨仓分桶解析；恢复策略归属及账本重建由调用方负责。"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class PositionBuckets:
    quantities: tuple
    costs: tuple


def position_buckets(rows, allowed_instruments, *, require_cost=False, unique_buckets=False):
    allowed = {str(i) for i in allowed_instruments}
    quantities, costs = {}, {}
    seen = set()
    for row in rows:
        try:
            qty = Decimal(str(row["Position"]))
            if not qty.is_finite() or qty < 0 or qty != qty.to_integral_value():
                raise ValueError("持仓数量须为非负整数")
            if not qty:
                continue
            key = row["InstrumentID"] + "." + row["ExchangeID"]
            direction, date = str(row.get("PosiDirection")), str(row.get("PositionDate"))
            if (key not in allowed or str(row.get("HedgeFlag")) != "1" or
                    direction not in {"2", "3"} or date not in {"1", "2"}):
                raise ValueError("合约、投机标志或今昨方向无效")
            index = (0 if direction == "2" else 2) + (0 if date == "1" else 1)
            if unique_buckets and (key, index) in seen:
                raise ValueError("今昨仓分桶重复")
            seen.add((key, index))
            cost = Decimal(str(row["PositionCost"])) if require_cost else Decimal(0)
            if require_cost and (not cost.is_finite() or cost <= 0):
                raise ValueError("持仓成本无效")
            quantities.setdefault(key, [Decimal(0)] * 4)[index] += qty
            costs.setdefault(key, [Decimal(0)] * 4)[index] += cost
        except (KeyError, ValueError, TypeError, InvalidOperation) as error:
            raise RuntimeError(f"权威今昨仓明细无效: {error}") from error
    return {key: PositionBuckets(tuple(value), tuple(costs[key])) for key, value in quantities.items()}
