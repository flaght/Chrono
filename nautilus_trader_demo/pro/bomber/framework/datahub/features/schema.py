"""声明式标量Schema生成不可变Payload；不导入配置中的类或执行表达式。"""
from dataclasses import make_dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import keyword
from threading import RLock

from .contracts import Alignment, DatasetSpec, FeatureError, FrequencySpec, PayloadField

_TYPES = {"decimal": Decimal, "float": float, "int": int, "str": str, "bool": bool}
_CACHE, _LOCK = {}, RLock()
_COMPARE = {"ge": lambda a, b: a >= b, "gt": lambda a, b: a > b,
    "le": lambda a, b: a <= b, "lt": lambda a, b: a < b}


def only_keys(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise FeatureError(f"{name}须为声明对象且不含未知配置键")


def dataset_from_definition(definition, *, calendars=None):
    only_keys(definition, ("dataset_id", "schema_version", "fields", "frequency", "alignment",
        "max_source_age_ns", "rejected_quality_flags", "computation_version", "relations"), "Schema")
    try:
        field_defs = definition["fields"]
        if not isinstance(field_defs, list) or not field_defs:
            raise FeatureError("Schema须有非空字段列表")
        fields, limits = [], []
        for item in field_defs:
            only_keys(item, ("name", "kind", "nullable", "minimum", "maximum", "exclusive_minimum"), "字段")
            name = item["name"]
            if type(name) is not str or name.startswith("_") or keyword.iskeyword(name):
                raise FeatureError("字段须为公开合法标识符")
            nullable = item.get("nullable", False)
            exclusive = item.get("exclusive_minimum", False)
            if type(nullable) is not bool or type(exclusive) is not bool:
                raise FeatureError("nullable/exclusive_minimum须为布尔值")
            field = PayloadField(name, item["kind"], nullable)
            fields.append(field)
            low, high = item.get("minimum"), item.get("maximum")
            if (low is not None or high is not None) and field.kind not in {"int", "decimal", "float"}:
                raise FeatureError("数值边界只适用于数值字段")
            def bound(value):
                if value is None:
                    return None
                if isinstance(value, bool):
                    raise FeatureError("数值边界不能为bool")
                result = Decimal(str(value))
                if not result.is_finite():
                    raise FeatureError("数值边界须为有限值")
                return result
            low, high = bound(low), bound(high)
            if exclusive and low is None or low is not None and high is not None and (low > high or exclusive and low == high):
                raise FeatureError("数值边界区间无效")
            limits.append((name, low, high, exclusive))
        kinds = {f.name: f.kind for f in fields}
        if len(kinds) != len(fields):
            raise FeatureError("Schema字段不能重复")
        relations = []
        for rule in definition.get("relations", ()):
            only_keys(rule, ("left", "op", "right"), "字段关系")
            left, op, right = rule["left"], rule["op"], rule["right"]
            if left not in kinds or right not in kinds or op not in _COMPARE or kinds[left] != kinds[right] \
                    or kinds[left] not in {"int", "decimal", "float"}:
                raise FeatureError("字段关系须声明同型数值字段和白名单比较操作")
            relations.append((left, op, right))
        # 类型身份覆盖整份Schema，阻止同名/同版本但频率或规则不同的记录混用。
        descriptor = definition
        canonical = json.dumps(descriptor, sort_keys=True, ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        def validate(self):
            for field in fields:
                field.validate(getattr(self, field.name))
            for name, low, high, exclusive in limits:
                value = getattr(self, name)
                if value is None:
                    continue
                value = Decimal(str(value)) if low is not None or high is not None else value
                if low is not None and (value < low or exclusive and value == low) or high is not None and value > high:
                    raise FeatureError(f"字段{name}超出声明数值边界")
            for left, op, right in relations:
                a, b = getattr(self, left), getattr(self, right)
                if a is not None and b is not None and not _COMPARE[op](a, b):
                    raise FeatureError(f"字段关系{left} {op} {right}不成立")
        with _LOCK:
            payload_type = _CACHE.get(canonical)
            if payload_type is None:
                payload_type = make_dataclass("Payload_" + digest[:20],
                    [(f.name, _TYPES[f.kind] | None if f.nullable else _TYPES[f.kind]) for f in fields],
                    frozen=True, slots=True, namespace={"__post_init__": validate, "__module__": __name__})
                _CACHE[canonical] = payload_type
        frequency = definition.get("frequency", {})
        only_keys(frequency, ("period_ns", "anchor_ns", "timezone", "calendar_id"), "频率")
        calendar_id = frequency.get("calendar_id", "continuous")
        slots = (calendars or {}).get(calendar_id)
        return DatasetSpec(definition["dataset_id"], definition["schema_version"], payload_type, tuple(fields),
            frequency=FrequencySpec(**frequency, slots=slots),
            default_alignment=Alignment(definition.get("alignment", "EXACT")),
            max_source_age_ns=definition.get("max_source_age_ns"),
            rejected_quality_flags=definition.get("rejected_quality_flags", 1),
            computation_version=definition.get("computation_version", "1") + ":declaration:" + digest)
    except (KeyError, TypeError, ValueError, InvalidOperation) as error:
        if isinstance(error, FeatureError):
            raise
        raise FeatureError("声明式Schema缺字段或类型/值无效") from None
