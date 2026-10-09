"""账户净额执行的策略归属账本；内部划转不是柜台成交。

归属按真实合约保存，策略查询汇总同一逻辑目标的新旧合约。外部成交只消费
发送前冻结的分配计划；当前目标变化不能重新解释已经发出的订单。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.trader.execution.contracts import OrderSide


class AttributionError(RuntimeError):
    """归属或订单关联不明确，不能继续净额执行。"""


class PositionAttributionLedger:
    def __init__(self, positions):
        self.positions = positions
        # (client_id, instrument_id, strategy_id, target_key) -> 有符号已成交量。
        self._owned = {}
        self._orders = {}
        self._order_index = {}
        self._active_tokens = set()
        self._transfers = []

    def state(self):
        with self.positions._lock:
            return deepcopy({
                "version": 1,
                "owned": [dict(client_id=k[0], instrument_id=k[1], strategy_id=k[2],
                    target_key=k[3], quantity=str(v)) for k, v in sorted(self._owned.items())],
                "orders": self._orders,
                "transfers": self._transfers,
            })

    def restore(self, state):
        if state and state.get("version") != 1:
            raise AttributionError("不支持的归属账本版本")
        owned = {}
        for row in state.get("owned", ()):
            key = tuple(row[n] for n in ("client_id", "instrument_id", "strategy_id", "target_key"))
            if key in owned or any(not isinstance(v, str) or not v.strip() for v in key):
                raise AttributionError("归属仓位键无效或重复")
            quantity = Decimal(row["quantity"])
            if not quantity.is_finite():
                raise AttributionError("归属仓位必须有限")
            owned[key] = quantity
        orders = deepcopy(state.get("orders", {}))
        for token, order in orders.items():
            total = Decimal(0)
            if order["direction"] not in (-1, 1):
                raise AttributionError("归属订单方向无效")
            for leg in order["legs"]:
                quantity, filled = Decimal(leg["quantity"]), Decimal(leg["filled"])
                if (not quantity.is_finite() or not filled.is_finite()
                        or quantity <= 0 or not 0 <= filled <= quantity):
                    raise AttributionError("归属订单数量无效")
                total += quantity
            if not token or total != Decimal(order["quantity"]):
                raise AttributionError("归属订单数量不守恒")
        with self.positions._lock:
            # 旧检查点没有真实合约归属，保持未归属，不凭逻辑仓位猜测迁移。
            totals = {}
            for key, quantity in owned.items():
                owner = key[2:]
                totals[owner] = totals.get(owner, Decimal(0)) + quantity
            if any(self.positions.position(*owner) != qty for owner, qty in totals.items()):
                raise AttributionError("归属明细与策略仓位不一致")
            self._owned, self._orders = owned, orders
            self._transfers = deepcopy(state.get("transfers", []))
            self._order_index = {}
            self._active_tokens = set()
            for token, order in orders.items():
                if order["client_order_id"] is not None:
                    key = (order["client_id"], order["client_order_id"])
                    if key in self._order_index:
                        raise AttributionError("多个归属计划绑定同一订单")
                    self._order_index[key] = token
                if any(amount for _, amount in self._remaining(order)):
                    self._active_tokens.add(token)

    def unassigned(self, client_id, instrument_id):
        with self.positions._lock:
            assigned = sum((v for k, v in self._owned.items()
                if k[:2] == (client_id, str(instrument_id))), Decimal(0))
            return self.positions.account_position(client_id, instrument_id) - assigned

    def adopt(self, client_id, instrument_id, strategy_id, target_key, quantity):
        quantity = Decimal(str(quantity))
        key = (client_id, str(instrument_id), strategy_id, target_key)
        with self.positions._lock:
            if (self.positions.working_quantity(client_id, instrument_id) != 0
                    or self.positions.account_position(client_id, instrument_id) != quantity
                    or any(k[:2] == key[:2] and v != 0 for k, v in self._owned.items())
                    or self.positions.position(strategy_id, target_key) != 0):
                raise AttributionError("接管要求无在途、归属为空且与权威仓位一致")
            self._owned[key] = quantity
            self.positions.set_strategy_position(strategy_id, target_key, quantity)

    def _adjust(self, key, delta):
        self._owned[key] = self._owned.get(key, Decimal(0)) + delta
        self.positions.adjust_strategy_position(key[2], key[3], delta)

    def _remaining(self, order):
        if order["released"]:
            return ()
        return ((leg, Decimal(leg["quantity"]) - Decimal(leg["filled"]))
                for leg in order["legs"])

    def prepare(self, request, orders):
        """风控通过后准备划转与订单分配；发送前再逐单登记预留量。"""
        rows = request.metadata.get("position_attribution")
        if rows is None:
            return orders  # 旧客户端直调保持显式未归属行为。
        with self.positions._lock:
            desired, totals = {}, {}
            for row in rows:
                key = (request.client_id, row["instrument_id"], row["strategy_id"], row["target_key"])
                quantity = Decimal(row["quantity"])
                if key in desired or not quantity.is_finite():
                    raise AttributionError("归属目标重复或不是有限数值")
                desired[key] = quantity
                totals[key[1]] = totals.get(key[1], Decimal(0)) + quantity
            targets = {str(k): Decimal(v) for k, v in request.targets.items()}
            if any(totals.get(k, Decimal(0)) != targets.get(k, Decimal(0))
                   for k in totals.keys() | targets.keys()):
                raise AttributionError("策略归属目标之和与账户目标不一致")
            owned_totals = {}
            for key, quantity in self._owned.items():
                owned_totals[key[2:]] = owned_totals.get(key[2:], Decimal(0)) + quantity
            for owner in {key[2:] for key in desired}:
                if self.positions.position(*owner) != owned_totals.get(owner, Decimal(0)):
                    raise AttributionError("逻辑归属缺少一致的真实合约明细，须先核验迁移")
            deficits = {k: desired.get(k, Decimal(0)) - self._owned.get(k, Decimal(0))
                for k in desired.keys() | self._owned.keys() if k[0] == request.client_id}
            reserved = {}
            for token in self._active_tokens:
                order = self._orders[token]
                if order["client_id"] != request.client_id:
                    continue
                for leg, remaining in self._remaining(order):
                    key = (request.client_id, order["instrument_id"], leg["strategy_id"], leg["target_key"])
                    signed = order["direction"] * remaining
                    deficits[key] = deficits.get(key, Decimal(0)) - signed
                    reserved[key[1]] = reserved.get(key[1], Decimal(0)) + signed
            instruments = set(targets) | {k[1] for k in deficits}
            for instrument in instruments:
                instrument_id = InstrumentId.from_str(instrument)
                if self.unassigned(request.client_id, instrument_id) != 0:
                    raise AttributionError(f"账户有未归属仓位，须先显式接管或对账: {instrument}")
                if self.positions.working_quantity(request.client_id, instrument_id) != reserved.get(instrument, 0):
                    raise AttributionError(f"账户在途量与归属订单预留不一致: {instrument}")
            transfers = []
            for instrument in sorted(instruments):
                positive = [k for k in sorted(deficits) if k[1] == instrument and deficits[k] > 0]
                negative = [k for k in sorted(deficits) if k[1] == instrument and deficits[k] < 0]
                for buy in positive:
                    for sell in negative:
                        quantity = min(deficits[buy], -deficits[sell])
                        if quantity <= 0:
                            continue
                        deficits[buy] -= quantity
                        deficits[sell] += quantity
                        transfers.append((buy, sell, quantity))
            attributed = []
            for order in orders:
                direction = 1 if order.side is OrderSide.BUY else -1
                remaining, legs = order.quantity, []
                for key in sorted(deficits):
                    if key[1] != str(order.instrument_id) or deficits[key] * direction <= 0:
                        continue
                    quantity = min(remaining, abs(deficits[key]))
                    if quantity:
                        legs.append(dict(strategy_id=key[2], target_key=key[3], quantity=str(quantity), filled="0"))
                        deficits[key] -= direction * quantity
                        remaining -= quantity
                if remaining:
                    raise AttributionError("规划订单超过可分配策略需求")
                attributed.append(replace(order, metadata={**order.metadata,
                    "allocation_token": uuid4().hex, "allocation_legs": tuple(legs)}))
            # 所有订单计划验证成功后再执行内部划转；不生成虚构的成交事件或费用。
            for buy, sell, quantity in transfers:
                self._adjust(buy, quantity)
                self._adjust(sell, -quantity)
                self._transfers.append(dict(client_id=request.client_id, instrument_id=buy[1],
                    buy_strategy=buy[2], buy_target=buy[3], sell_strategy=sell[2], sell_target=sell[3],
                    quantity=str(quantity), ts_event=request.ts_event, revision=request.revision))
            return tuple(attributed)

    def reserve(self, order):
        token = order.metadata.get("allocation_token")
        if token is None:
            return
        with self.positions._lock:
            if token in self._orders:
                raise AttributionError("归属计划ID重复")
            self._orders[token] = dict(client_id=order.backend_id, instrument_id=str(order.instrument_id),
                submitting_strategy=order.strategy_id, direction=1 if order.side is OrderSide.BUY else -1,
                quantity=str(order.quantity), client_order_id=None, released=False,
                legs=deepcopy(list(order.metadata["allocation_legs"])))
            self._active_tokens.add(token)

    def bind(self, client_order_id, order):
        token = order.metadata.get("allocation_token")
        if token is not None:
            with self.positions._lock:
                key = (order.backend_id, client_order_id)
                previous = self._order_index.get(key)
                if ((previous is not None and previous != token)
                        or self._orders[token]["client_order_id"] not in (None, client_order_id)):
                    raise AttributionError("同一订单绑定不同归属计划")
                self._orders[token]["client_order_id"] = client_order_id
                self._order_index[key] = token

    def associate_report(self, report):
        """先关联可信token，使并发检查点能发现订单状态与归属更新的中间态。"""
        token = report.metadata.get("allocation_token")
        if not isinstance(token, str):
            return
        with self.positions._lock:
            order = self._orders.get(token)
            if (order is not None and order["client_id"] == report.backend_id
                    and order["instrument_id"] == str(report.instrument_id)
                    and order["submitting_strategy"] == report.metadata.get("strategy_id")
                    and order["client_order_id"] in (None, report.client_order_id)):
                key = (report.backend_id, report.client_order_id)
                if self._order_index.get(key, token) == token:
                    order["client_order_id"] = report.client_order_id
                    self._order_index[key] = token

    def release_unsent(self, order):
        token = order.metadata.get("allocation_token")
        if token is not None:
            with self.positions._lock:
                record = self._orders[token]
                if any(Decimal(leg["filled"]) for leg in record["legs"]):
                    raise AttributionError("已有成交的订单不能解释为未发送")
                record["released"] = True
                self._active_tokens.discard(token)

    def apply(self, report, update, *, account_id=None):
        """只接受状态机提供的增量；返回含归属事实的报告供回调分发。"""
        with self.positions._lock:
            token = report.metadata.get("allocation_token")
            if token is not None and not isinstance(token, str):
                raise AttributionError("回报归属计划ID必须为字符串")
            known = self._order_index.get((report.backend_id, report.client_order_id))
            if token is None:
                token = known
            if token is None:
                if any(row["client_id"] == report.backend_id
                       and row["instrument_id"] == str(report.instrument_id)
                       and not row["released"]
                       and any(amount for _, amount in self._remaining(row))
                       for row in (self._orders[key] for key in self._active_tokens)):
                    raise AttributionError("存在归属活动订单，但回报缺少可信订单关联")
                return replace(report, metadata={**report.metadata,
                    "attribution_status": "UNASSIGNED"})
            order = self._orders.get(token)
            if (order is None or (known is not None and known != token)
                    or report.metadata.get("account_id", account_id) != account_id
                    or order["client_id"] != report.backend_id
                    or order["instrument_id"] != str(report.instrument_id)
                    or order["direction"] != (1 if update.state.side is OrderSide.BUY else -1)
                    or Decimal(order["quantity"]) != update.state.order_quantity
                    or report.metadata.get("strategy_id") != order["submitting_strategy"]
                    or order["client_order_id"] not in (None, report.client_order_id)):
                raise AttributionError("成交回报与发送前归属关联不一致")
            if order["released"] and update.fill_delta:
                raise AttributionError("已释放的归属订单仍收到新增成交")
            if sum((Decimal(leg["filled"]) for leg in order["legs"]), Decimal(0)) + update.fill_delta != update.state.filled_quantity:
                raise AttributionError("订单状态机与归属累计成交不一致")
            remaining, fills = update.fill_delta, []
            planned = []
            for leg, capacity in self._remaining(order):
                quantity = min(remaining, capacity)
                if quantity:
                    planned.append((leg, quantity))
                    remaining -= quantity
            if remaining:
                raise AttributionError("成交超过归属计划剩余数量")
            order["client_order_id"] = report.client_order_id
            self._order_index[(report.backend_id, report.client_order_id)] = token
            for leg, quantity in planned:
                leg["filled"] = str(Decimal(leg["filled"]) + quantity)
                key = (report.backend_id, str(report.instrument_id), leg["strategy_id"], leg["target_key"])
                self._adjust(key, order["direction"] * quantity)
                fills.append(dict(strategy_id=leg["strategy_id"], target_key=leg["target_key"],
                    quantity=str(quantity), cumulative_filled=leg["filled"]))
            if update.release_delta:
                order["released"] = True
            if not any(amount for _, amount in self._remaining(order)):
                self._active_tokens.discard(token)
            return replace(report, metadata={**report.metadata, "allocation_token": token,
                "allocation_owners": tuple(dict(strategy_id=leg["strategy_id"], target_key=leg["target_key"])
                    for leg in order["legs"]), "attributed_fills": tuple(fills)})

    @staticmethod
    def validate_checkpoints(state, checkpoints):
        """已关联订单必须和订单状态机同代；不把冲突检查点带入重启。"""
        machines = {client: {item.state.client_order_id: item.state for item in items}
            for client, items in checkpoints.items()}
        for order in state.get("orders", {}).values():
            client, order_id = order["client_id"], order["client_order_id"]
            if client not in machines:
                raise AttributionError("归属检查点缺少对应客户端的订单状态机")
            if order_id is None:
                continue
            local = machines[client].get(order_id)
            filled = sum((Decimal(leg["filled"]) for leg in order["legs"]), Decimal(0))
            if local is None and order["released"] and not filled:
                continue  # 发送失败时待确认记录已经撤回。
            if (local is None or str(local.instrument_id) != order["instrument_id"]
                    or local.order_quantity != Decimal(order["quantity"])
                    or local.filled_quantity != filled
                    or (1 if local.side is OrderSide.BUY else -1) != order["direction"]
                    or (order["released"] and not local.status.is_terminal)):
                raise AttributionError("归属计划与订单状态机检查点不一致")
