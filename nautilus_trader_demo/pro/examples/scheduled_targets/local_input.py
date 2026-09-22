"""本地 CSV 到 DataHub 目标计划的示例适配器；核心不绑定存储格式。"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from datahub.target_schedule import TargetPlan, TargetScheduleStore


def _parse_timestamp(raw: str) -> datetime:
    normalized = raw.strip().replace("/", "-")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(normalized, pattern)
            except ValueError:
                continue
        raise ValueError(f"无法解析目标时点: {raw!r}")


def _to_ns(stamp: datetime) -> int:
    delta = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000
            + delta.microseconds) * 1_000


def load_target_csv(path: str | Path, *, timezone: str = "Asia/Shanghai") -> TargetScheduleStore:
    """读取目标CSV；支持本框架target_key及旧文件instrument_id两种列名。"""
    groups: dict[int, dict[str, Decimal]] = defaultdict(dict)
    instrument_types: dict[int, dict[str, str]] = defaultdict(dict)
    zone = ZoneInfo(timezone)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        key_column = "target_key" if "target_key" in fields else "instrument_id"
        required = {"timestamp", key_column, "target_qty"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"目标文件缺少列: {sorted(required - set(reader.fieldnames or ()))}")
        for line, row in enumerate(reader, start=2):
            try:
                stamp = _parse_timestamp(row["timestamp"])
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=zone)
                slot = _to_ns(stamp)
                key = row[key_column].strip()
                quantity = Decimal(row["target_qty"].strip())
            except (ValueError, KeyError, InvalidOperation, AttributeError) as exc:
                raise ValueError(f"目标文件第{line}行无效: {exc}") from exc
            if not key or not quantity.is_finite():
                raise ValueError(f"目标文件第{line}行目标键或数量无效")
            if key in groups[slot]:
                raise ValueError(f"目标文件第{line}行重复: {slot}/{key}")
            groups[slot][key] = quantity
            if "instrument_type" in fields:
                kind = (row["instrument_type"] or "").strip().upper()
                if not kind:
                    raise ValueError(f"目标文件第{line}行资产类型为空")
                instrument_types[slot][key] = kind
    if not groups:
        raise ValueError("目标文件为空")
    return TargetScheduleStore(tuple(
        TargetPlan(slot, 0, targets, source_revision=str(Path(path).name),
                   metadata={"instrument_types": tuple(sorted(instrument_types[slot].items()))}
                   if instrument_types[slot] else {})
        for slot, targets in groups.items()
    ))
