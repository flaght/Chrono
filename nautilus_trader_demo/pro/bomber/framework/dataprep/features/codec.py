"""CSV/Parquet/数据库共用显式列映射；不推测字段、时间单位或发布时间。"""
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
import hashlib
import json
from numbers import Integral, Real
from types import MappingProxyType

from bomber.framework.datahub.features import ExternalDataRecord, FeatureError

ENVELOPE_COLUMNS = (
    "dataset_id", "schema_version", "key", "ts_event", "available_ns", "source_ns",
    "revision", "source", "quality_flags",
)


@dataclass(frozen=True)
class FeatureColumnMapping:
    columns: object = None  # 标准列名 -> 实际列名；payload字段使用payload.<name>。
    timestamp_label: str = "end"

    def __post_init__(self):
        columns = dict(self.columns or {})
        if self.timestamp_label not in {"start", "end", "inclusive-end"} or any(
            type(key) is not str or type(value) is not str or not value for key, value in columns.items()
        ):
            raise FeatureError("列映射/时间标签无效")
        if len(set(columns.values())) != len(columns):
            raise FeatureError("一个实际字段不能映射多个标准字段")
        object.__setattr__(self, "columns", MappingProxyType(columns))

    def column(self, canonical):
        return self.columns.get(canonical, canonical)


def _integer(value, name):
    if isinstance(value, bool):
        raise FeatureError(f"{name}不能为bool")
    if isinstance(value, Integral):
        return int(value)
    # CSV文本可解析；float会丢纳秒精度，因此即使为整数也拒绝。
    if type(value) is str and value.isascii() and value.isdigit():
        return int(value)
    raise FeatureError(f"{name}须为整数或整数字符串，时间单位固定UTC Unix纳秒")


def _payload_value(value, field):
    if value is None or type(value) is str and value == "":
        if field.nullable:
            return None
        raise FeatureError(f"字段{field.name}不可空")
    try:
        if field.kind == "int":
            if type(value) is str and value.startswith("-"):
                result = -_integer(value[1:], field.name)
            else:
                result = _integer(value, field.name)
        elif field.kind == "decimal":
            result = Decimal(str(value))
        elif field.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (str, Real, Decimal)):
                raise FeatureError(f"字段{field.name}须为数值")
            result = float(value)
        elif field.kind == "bool":
            if type(value) is bool:
                result = value
            elif type(value) is str and value in {"true", "false"}:
                result = value == "true"
            else:
                raise FeatureError(f"字段{field.name}须为true/false")
        else:
            if type(value) is not str:
                raise FeatureError(f"字段{field.name}须为字符串")
            result = value
    except (ValueError, TypeError, InvalidOperation):
        raise FeatureError(f"字段{field.name}无法解析为{field.kind}") from None
    field.validate(result)
    return result


def decode_record(row, registry, mapping=None):
    mapping = mapping or FeatureColumnMapping()
    def required(name):
        column = mapping.column(name)
        if column not in row or row[column] is None:
            raise FeatureError(f"缺少必需字段{name}（实际列{column}）")
        return row[column]
    dataset, key, source = (required(name) for name in ("dataset_id", "key", "source"))
    if any(type(value) is not str or not value for value in (dataset, key, source)):
        raise FeatureError("数据集、业务键、来源须为非空字符串；指数代码保留前导零")
    version = _integer(required("schema_version"), "schema_version")
    spec = registry.get(dataset, version)
    expected_payload = {mapping.column("payload." + f.name) for f in spec.payload_fields}
    if any(type(name) is str and name.startswith("payload.") and name not in expected_payload for name in row):
        raise FeatureError("出现未注册Payload字段")
    integers = {name: _integer(required(name), name) for name in
        ("ts_event", "available_ns", "source_ns", "revision", "quality_flags")}
    integers["ts_event"] = spec.frequency.close_ns(integers["ts_event"], mapping.timestamp_label)
    received = row.get(mapping.column("received_ns"))
    def payload_value(field):
        name = "payload." + field.name
        if mapping.column(name) not in row:
            raise FeatureError(f"缺少必需字段{name}")
        return _payload_value(row[mapping.column(name)], field)
    payload = spec.payload_type(**{field.name: payload_value(field)
        for field in spec.payload_fields})
    return ExternalDataRecord(dataset, version, key, payload, source=source,
        received_ns=None if received in (None, "") else _integer(received, "received_ns"), **integers)


def encode_record(record):
    row = {name: getattr(record, name) for name in ENVELOPE_COLUMNS}
    row["received_ns"] = record.received_ns
    for field in fields(record.payload):
        value = getattr(record.payload, field.name)
        row["payload." + field.name] = str(value) if isinstance(value, Decimal) else value
    return row


def record_digest(record):
    row = encode_record(record)
    row.pop("received_ns")  # 本地接收证据不改变上游版本内容。
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def mapping_digest(mapping):
    descriptor = (sorted(mapping.columns.items()), mapping.timestamp_label)
    return hashlib.sha256(json.dumps(descriptor, ensure_ascii=False).encode()).hexdigest()
