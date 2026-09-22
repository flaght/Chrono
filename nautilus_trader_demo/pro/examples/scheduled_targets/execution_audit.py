"""第六类回测的逐时点执行审计；只读取标准回报，不参与下单。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from datahub.target_schedule import TargetScheduleStore


@dataclass(frozen=True)
class AuditedFill:
    slot_ns: int
    fill_ns: int
    instrument_id: str
    side: str
    quantity: Decimal
    price: Decimal
    commission: Decimal
    commission_currency: str
    client_order_id: str


@dataclass(frozen=True)
class ScheduleExecutionAudit:
    fills: tuple[AuditedFill, ...]
    final_targets: Mapping[str, Decimal]
    commission_by_currency: Mapping[str, Decimal]


def audit_schedule_execution(
    schedule: TargetScheduleStore,
    reports: Sequence[Any],
    instrument_by_key: Mapping[str, Any],
    *,
    actual_positions: Mapping[str, Decimal],
    working_positions: Mapping[str, Decimal],
    native_positions: Mapping[str, Decimal],
) -> ScheduleExecutionAudit:
    """对计划版本、严格下一时点成交、每时点仓差、费用及终态仓位作 fail-closed 验证。"""
    if not schedule.plans:
        raise AssertionError("执行审计需要至少一个目标计划")
    known_slots = set(schedule.slots)
    symbols = {key: str(instrument) for key, instrument in instrument_by_key.items()}
    missing_routes = {key for plan in schedule.plans for key in plan.targets} - set(symbols)
    if missing_routes:
        raise AssertionError(f"计划目标缺少执行路由: {sorted(missing_routes)}")
    fills: list[AuditedFill] = []
    fees: dict[str, Decimal] = defaultdict(Decimal)
    actual_deltas: dict[int, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    for report in reports:
        kind = str(getattr(report.report_type, "value", report.report_type))
        if kind in {"REJECTED", "CANCELED"}:
            raise AssertionError(f"执行回报未完成: {kind} {report.client_order_id}")
        if kind not in {"FILLED", "PARTIALLY_FILLED"}:
            continue
        slot = report.metadata.get("schedule_slot_ns")
        if slot is None or int(slot) not in known_slots:
            raise AssertionError(f"成交缺少有效目标时点: {report.client_order_id}")
        slot = int(slot)
        if report.ts_event <= slot:
            raise AssertionError(f"成交早于或等于目标时点: {report.client_order_id} "
                                 f"slot={slot} fill={report.ts_event}")
        next_slots = tuple(item for item in schedule.slots if item > slot)
        if next_slots and report.ts_event >= next_slots[0]:
            raise AssertionError(f"前一时点订单跨越下一计划时点仍在成交: {report.client_order_id}")
        symbol = str(report.instrument_id)
        if symbol not in symbols.values():
            raise AssertionError(f"成交来自计划外合约: {symbol}")
        side = str(getattr(report.order_side, "value", report.order_side))
        quantity = Decimal(str(report.filled_quantity))
        price = None if report.fill_price is None else Decimal(str(report.fill_price))
        if side not in {"BUY", "SELL"} or quantity <= 0 or price is None or price <= 0:
            raise AssertionError(f"成交方向、数量或价格无效: {report.client_order_id}")
        fee_raw = report.metadata.get("commission")
        if fee_raw is None:
            raise AssertionError(f"成交缺少手续费: {report.client_order_id}")
        parts = str(fee_raw).replace("_", "").split()
        try:
            fee = Decimal(parts[0])
        except (InvalidOperation, IndexError) as exc:
            raise AssertionError(f"手续费格式无效: {fee_raw}") from exc
        if not fee.is_finite() or fee < 0:
            raise AssertionError(f"手续费金额无效: {fee_raw}")
        currency = parts[1] if len(parts) > 1 else "UNSPECIFIED"
        fills.append(AuditedFill(slot, report.ts_event, symbol, side, quantity,
                                 price, fee, currency, report.client_order_id))
        actual_deltas[slot][symbol] += quantity if side == "BUY" else -quantity
        fees[currency] += fee
    if not fills:
        raise AssertionError("没有可审计的成交回报")
    previous = {symbol: Decimal(0) for symbol in symbols.values()}
    for plan in schedule.plans:
        desired = {symbol: Decimal(0) for symbol in symbols.values()}
        desired.update({symbols[key]: quantity for key, quantity in plan.targets.items()})
        for symbol in symbols.values():
            expected_delta = desired[symbol] - previous[symbol]
            observed_delta = actual_deltas[plan.slot_ns].get(symbol, Decimal(0))
            if observed_delta != expected_delta:
                raise AssertionError(f"{plan.slot_ns}/{symbol}成交净变仓错误: "
                                     f"expected={expected_delta} observed={observed_delta}")
        previous = desired
    for label, actual in (("统一仓位", actual_positions), ("原生仓位", native_positions)):
        for symbol, target in previous.items():
            if Decimal(str(actual.get(symbol, 0))) != target:
                raise AssertionError(f"{label}不等于最终目标: {symbol} "
                                     f"actual={actual.get(symbol, 0)} target={target}")
    if any(Decimal(str(working_positions.get(symbol, 0))) != 0 for symbol in previous):
        raise AssertionError(f"仍有在途数量: {working_positions}")
    return ScheduleExecutionAudit(tuple(fills), previous, dict(fees))


def format_audit(audit: ScheduleExecutionAudit, *, timezone: str = "Asia/Shanghai") -> tuple[str, ...]:
    zone = ZoneInfo(timezone)
    rows = []
    for fill in audit.fills:
        slot = datetime.fromtimestamp(fill.slot_ns / 1_000_000_000, zone)
        executed = datetime.fromtimestamp(fill.fill_ns / 1_000_000_000, zone)
        rows.append(
            f"slot={slot:%Y-%m-%d %H:%M:%S} fill={executed:%Y-%m-%d %H:%M:%S} "
            f"instrument={fill.instrument_id} side={fill.side} qty={fill.quantity} "
            f"price={fill.price} fee={fill.commission} {fill.commission_currency} "
            f"order={fill.client_order_id}"
        )
    return tuple(rows)
