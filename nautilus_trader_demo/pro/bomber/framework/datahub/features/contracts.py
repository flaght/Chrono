"""外部数据的强类型协议；不依赖 SDK、数据库或策略执行。"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from enum import Enum
import hashlib
import json
import math
from typing import Callable
from zoneinfo import ZoneInfo

MINUTE_NS = 60_000_000_000


class FeatureError(ValueError):
    """Schema、时间或版本冲突；不能按缺数据降级。"""


class FeatureUnavailable(LookupError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class Alignment(str, Enum):
    EXACT = "EXACT"
    WINDOW = "WINDOW"
    ASOF = "ASOF"


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise FeatureError(f"{name}须为大于等于{minimum}的整数")
    return value


@dataclass(frozen=True)
class FrequencySpec:
    period_ns: int = MINUTE_NS
    anchor_ns: int = 0
    timezone: str = "Asia/Shanghai"
    calendar_id: str = "continuous"
    # 交易日历实现注入预期收盘槽；不在数据消息里携带可执行对象。
    slots: Callable[[int, int], tuple[int, ...]] | None = None

    def __post_init__(self):
        integer(self.period_ns, "period_ns", 1)
        integer(self.anchor_ns, "anchor_ns")
        ZoneInfo(self.timezone)
        if not self.calendar_id or self.calendar_id != "continuous" and not callable(self.slots):
            raise FeatureError("交易日历频率须显式提供slots接口")

    def close_ns(self, timestamp, label="end"):
        integer(timestamp, "timestamp")
        if label == "inclusive-end":
            timestamp += 1
        elif label == "start":
            timestamp += self.period_ns
        elif label != "end":
            raise FeatureError("时间标签须为start/end/inclusive-end")
        if (timestamp - self.anchor_ns) % self.period_ns:
            raise FeatureError("时间未对齐数据集频率")
        return timestamp

    def expected_ends(self, start_ns, end_ns):
        if start_ns >= end_ns:
            raise FeatureError("窗口须为非空(start,end]收盘槽区间")
        if self.slots is not None:
            result = tuple(self.slots(start_ns, end_ns))
        else:
            first = ((start_ns - self.anchor_ns) // self.period_ns + 1) * self.period_ns + self.anchor_ns
            if (end_ns - first) // self.period_ns > 1_000_000:
                raise FeatureError("窗口槽数量过大")
            result = tuple(range(first, end_ns + 1, self.period_ns))
        if not result or tuple(sorted(set(result))) != result or any(
            type(slot) is not int or not start_ns < slot <= end_ns
            or (slot - self.anchor_ns) % self.period_ns for slot in result
        ):
            raise FeatureError("日历必须返回有序、不重复、频率匹配的窗口收盘槽")
        return result


@dataclass(frozen=True)
class PayloadField:
    name: str
    kind: str
    nullable: bool = False

    def __post_init__(self):
        if not self.name.isidentifier() or self.kind not in {"float", "int", "decimal", "str", "bool"}:
            raise FeatureError("Payload字段须有合法名称及已支持类型")

    def validate(self, value):
        if value is None and self.nullable:
            return
        valid = {
            "float": lambda: type(value) is float and math.isfinite(value),
            "int": lambda: type(value) is int,
            "decimal": lambda: isinstance(value, Decimal) and value.is_finite(),
            "str": lambda: type(value) is str,
            "bool": lambda: type(value) is bool,
        }[self.kind]()
        if not valid:
            raise FeatureError(f"字段{self.name}不符合{self.kind}类型/有限值要求")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    schema_version: int
    payload_type: type
    payload_fields: tuple[PayloadField, ...]
    frequency: FrequencySpec = FrequencySpec()
    default_alignment: Alignment = Alignment.EXACT
    max_source_age_ns: int | None = None
    rejected_quality_flags: int = 1
    computation_version: str = "1"

    def __post_init__(self):
        if any(type(value) is not str or not value for value in (self.dataset_id, self.computation_version)):
            raise FeatureError("数据集和计算版本不能为空")
        integer(self.schema_version, "schema_version", 1)
        integer(self.rejected_quality_flags, "rejected_quality_flags")
        if self.max_source_age_ns is not None:
            integer(self.max_source_age_ns, "max_source_age_ns")
        if not isinstance(self.frequency, FrequencySpec) or not isinstance(self.default_alignment, Alignment):
            raise FeatureError("频率/对齐须为声明的类型")
        params = getattr(self.payload_type, "__dataclass_params__", None)
        if not isinstance(self.payload_type, type) or not is_dataclass(self.payload_type) or not params.frozen:
            raise FeatureError("Payload类型须为已注册的frozen dataclass")
        names = tuple(field.name for field in self.payload_fields)
        if len(set(names)) != len(names) or set(names) != {field.name for field in fields(self.payload_type)}:
            raise FeatureError("Payload字段必须与类型完全一致")
        object.__setattr__(self, "payload_fields", tuple(self.payload_fields))

    @property
    def fingerprint(self):
        descriptor = (self.dataset_id, self.schema_version, self.computation_version,
            [(f.name, f.kind, f.nullable) for f in self.payload_fields],
            self.frequency.period_ns, self.frequency.anchor_ns, self.frequency.timezone,
            self.frequency.calendar_id, self.default_alignment.value,
            self.max_source_age_ns, self.rejected_quality_flags)
        return hashlib.sha256(json.dumps(descriptor, ensure_ascii=False).encode()).hexdigest()

    def validate_payload(self, payload):
        if type(payload) is not self.payload_type:
            raise FeatureError("Payload类型与注册Schema不一致")
        for field in self.payload_fields:
            field.validate(getattr(payload, field.name))


class DatasetRegistry:
    def __init__(self, specs=()):
        self._specs = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec):
        if not isinstance(spec, DatasetSpec):
            raise FeatureError("只能注册DatasetSpec")
        identity = (spec.dataset_id, spec.schema_version)
        if identity in self._specs:
            raise FeatureError("数据集Schema版本已注册，不允许原地修改")
        self._specs[identity] = spec

    def get(self, dataset_id, schema_version):
        try:
            return self._specs[(dataset_id, schema_version)]
        except KeyError:
            raise FeatureError(f"未注册Schema: {dataset_id}/{schema_version}") from None


@dataclass(frozen=True)
class ExternalDataRecord:
    dataset_id: str
    schema_version: int
    key: str
    payload: object
    ts_event: int  # 规范化业务区间结束边界；区间为[start,end)。
    available_ns: int
    source_ns: int
    revision: int
    source: str
    quality_flags: int = 0
    received_ns: int | None = None

    def __post_init__(self):
        if any(type(value) is not str or not value for value in (self.dataset_id, self.key, self.source)):
            raise FeatureError("数据集、键、来源不能为空")
        for name in ("ts_event", "available_ns", "source_ns", "revision", "quality_flags"):
            integer(getattr(self, name), name)
        integer(self.schema_version, "schema_version", 1)
        if self.received_ns is not None:
            integer(self.received_ns, "received_ns")
        if self.source_ns > self.available_ns or self.available_ns < self.ts_event:
            raise FeatureError("业务区间结束和原始观测均不得晚于实际发布时间")

    @property
    def visible_ns(self):
        return max(self.available_ns, self.received_ns or 0)

    @property
    def ts_init(self):
        """目标架构ts_init对应真实available_ns；接收时间另存，不覆盖发布时间。"""
        return self.available_ns

    @property
    def identity(self):
        return (self.dataset_id, self.schema_version, self.key, self.ts_event, self.revision)


@dataclass(frozen=True)
class FeatureRequirement:
    alias: str
    dataset_id: str
    schema_version: int
    key: str
    alignment: Alignment | None = None
    window_ns: int | None = None
    max_source_age_ns: int | None = None

    def __post_init__(self):
        if any(type(value) is not str or not value for value in (self.alias, self.dataset_id, self.key)):
            raise FeatureError("特征需求须有alias、数据集与显式业务键")
        integer(self.schema_version, "schema_version", 1)
        if self.alignment is not None and not isinstance(self.alignment, Alignment):
            raise FeatureError("对齐方式须为Alignment")
        if self.window_ns is not None:
            integer(self.window_ns, "window_ns", 1)
        if self.max_source_age_ns is not None:
            integer(self.max_source_age_ns, "max_source_age_ns")


@dataclass(frozen=True)
class FeatureQuery:
    event_ns: int
    as_of_ns: int

    def __post_init__(self):
        integer(self.event_ns, "event_ns")
        integer(self.as_of_ns, "as_of_ns")
        if self.event_ns > self.as_of_ns:
            raise FeatureError("业务截止点不得晚于决策可见时点")


@dataclass(frozen=True)
class MinuteSignal:
    score: float


@dataclass(frozen=True)
class IndexPrice:
    close: Decimal

    def __post_init__(self):
        if not isinstance(self.close, Decimal) or not self.close.is_finite() or self.close <= 0:
            raise FeatureError("指数价格须为有限正Decimal，不修改输入价格")


@dataclass(frozen=True)
class CommoditySpot:
    price: Decimal
    currency: str
    unit: str
    region: str
    grade: str

    def __post_init__(self):
        if not isinstance(self.price, Decimal) or not self.price.is_finite() or self.price <= 0:
            raise FeatureError("现货价格须为有限正Decimal")
        if any(type(value) is not str or not value for value in (self.currency, self.unit, self.region, self.grade)):
            raise FeatureError("现货须声明币种、计价单位、地区和品级")
