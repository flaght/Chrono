"""CSV 到不可变目标计划的本地适配器。"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from datahub.target_schedule import TargetPlan, TargetScheduleStore


def _timestamp_ns(raw: str, zone: ZoneInfo) -> int:
    try:
        stamp = datetime.fromisoformat(raw.strip().replace("/", "-"))
    except ValueError as exc:
        raise ValueError(f"无法解析目标时点: {raw!r}") from exc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=zone)
    delta = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000
            + delta.microseconds) * 1_000


def load_target_csv(path: Path, *, timezone_name: str = "Asia/Shanghai") -> TargetScheduleStore:
    """同一 timestamp 的所有行是一份完整组合，未列出的旧目标会被清零。"""
    groups: dict[int, dict[str, Decimal]] = defaultdict(dict)
    zone = ZoneInfo(timezone_name)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        key_column = "target_key" if "target_key" in fields else "instrument_id"
        required = {"timestamp", key_column, "target_qty"}
        if required - fields:
            raise ValueError(f"目标文件缺少列: {sorted(required - fields)}")
        for line, row in enumerate(reader, 2):
            try:
                slot = _timestamp_ns(row["timestamp"], zone)
                key = row[key_column].strip()
                quantity = Decimal(row["target_qty"].strip())
            except (KeyError, AttributeError, ValueError, InvalidOperation) as exc:
                raise ValueError(f"目标文件第{line}行无效: {exc}") from exc
            if not key or not quantity.is_finite() or quantity != quantity.to_integral_value():
                raise ValueError(f"目标文件第{line}行目标键或手数无效")
            if "instrument_type" in fields:
                kind = str(row["instrument_type"] or "").strip().upper()
                if kind not in {"FUTURE", "FUTURES", "CTP_FUTURES"}:
                    raise ValueError(f"目标文件第{line}行仅支持真实期货目标: {kind}")
            if key in groups[slot]:
                raise ValueError(f"目标文件第{line}行重复目标: {key}")
            groups[slot][key] = quantity
    if not groups:
        raise ValueError("目标文件为空")
    return TargetScheduleStore(tuple(
        TargetPlan(slot, 0, targets, source_revision=path.name)
        for slot, targets in groups.items()
    ))
