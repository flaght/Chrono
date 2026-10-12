"""DolphinDB原表声明式投影；不认识任何特征名称、固定频率或业务键格式。"""
from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
from threading import RLock
from types import MappingProxyType
import re
import time
from zoneinfo import ZoneInfo

from bomber.framework.datahub.features import ExternalDataRecord, FeatureError
from bomber.framework.datahub.features.schema import only_keys
from .base import FeatureProvider, FeatureSourceError, select_records
from .codec import _integer, _payload_value
from .database import identifier
from .dolphin_query import DolphinDbReadSession


def literal(value):
    if type(value) is not str or any(ord(c) < 32 or c in {'"', "\\"} for c in value):
        raise FeatureError("业务键须为字符串且不含控制字符、引号或反斜线")
    return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class RawTimestamp:
    column: str
    date_column: str | None = None
    encoding: str = "dolphindb"
    timezone: str = "Asia/Shanghai"
    label: str = "end"

    def __post_init__(self):
        identifier(self.column)
        if self.date_column is not None:
            identifier(self.date_column)
        if self.encoding not in {"dolphindb", "unix_ns"} or self.timezone not in {"UTC", "Asia/Shanghai"} \
                or self.label not in {"start", "end", "inclusive-end"}:
            raise FeatureError("原表时间须明确编码、时区及业务标签")
        if self.encoding == "unix_ns" and (self.date_column is not None or self.timezone != "UTC"):
            raise FeatureError("unix_ns须为UTC整数，无date拼接")

    @classmethod
    def from_definition(cls, value):
        only_keys(value, ("column", "date_column", "encoding", "timezone", "label"), "时间映射")
        return cls(**value)


@dataclass(frozen=True)
class DolphinDbRawConfig:
    connection: object
    dataset_id: str
    schema_version: int
    database: str
    table: str
    key_column: str
    event_time: RawTimestamp
    payload_columns: object
    publication_time: RawTimestamp | None = None
    partition_date: str | None = None
    availability: str = "observed"
    revision_column: str | None = None
    quality_column: str | None = None

    def __post_init__(self):
        from ..sources.dolphindb import DolphinDbReferenceConfig
        if not isinstance(self.connection, DolphinDbReferenceConfig):
            raise FeatureError("原表连接须为DolphinDbReferenceConfig")
        if type(self.database) is not str or re.fullmatch(r"dfs://[A-Za-z0-9_/-]+", self.database) is None:
            raise FeatureError("原表数据库须为安全DFS路径")
        for name in (self.table, self.key_column, self.partition_date, self.revision_column, self.quality_column):
            if name is not None:
                identifier(name)
        columns = dict(self.payload_columns)
        if not columns:
            raise FeatureError("须显式映射Payload字段")
        for name, column in columns.items():
            identifier(name)
            identifier(column)
        if not isinstance(self.event_time, RawTimestamp) or self.publication_time is not None \
                and not isinstance(self.publication_time, RawTimestamp):
            raise FeatureError("时间映射须为RawTimestamp")
        if self.publication_time is not None and self.publication_time.label != "end":
            raise FeatureError("发布时间不能加业务频率偏移")
        if self.availability not in {"observed", "source-time"}:
            raise FeatureError("可见性须为observed或source-time")
        if self.availability == "source-time" and self.publication_time is None:
            raise FeatureError("source-time须有已确认的发布时间映射")
        if self.availability == "observed" and self.revision_column is not None:
            raise FeatureError("observed版本以真实观测生成，不能混用源修订序号")
        object.__setattr__(self, "payload_columns", MappingProxyType(columns))

    @property
    def timestamp_label(self):
        return self.event_time.label

    @property
    def identity(self):
        from dataclasses import asdict
        descriptor = (self.dataset_id, self.schema_version, sorted(self.payload_columns.items()), self.key_column,
            asdict(self.event_time), asdict(self.publication_time) if self.publication_time else None,
            self.partition_date, self.availability, self.revision_column, self.quality_column)
        digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        c = self.connection
        return f"dolphindb-raw:{c.host}:{c.port}:{self.database}/{self.table}:{digest}"

    @classmethod
    def from_definition(cls, value, connection, spec):
        only_keys(value, ("backend", "database", "table", "key_column", "event_time", "payload_columns",
            "publication_time", "partition_date", "availability", "revision_column", "quality_column"), "原表来源")
        if value.get("backend") != "dolphindb-raw":
            raise FeatureError("此来源定义须选择dolphindb-raw格式")
        options = {key: item for key, item in value.items() if key != "backend"}
        options["event_time"] = RawTimestamp.from_definition(options["event_time"])
        if options.get("publication_time") is not None:
            options["publication_time"] = RawTimestamp.from_definition(options["publication_time"])
        return cls(connection, spec.dataset_id, spec.schema_version, **options)


class DolphinDbRawFeatureProvider(FeatureProvider):
    backend = "dolphindb-raw"

    def __init__(self, config, registry, *, session_factory=None, clock_ns=None):
        super().__init__(registry)
        if not isinstance(config, DolphinDbRawConfig):
            raise FeatureError("原表Provider须使用DolphinDbRawConfig")
        self.spec = registry.get(config.dataset_id, config.schema_version)
        if set(config.payload_columns) != {f.name for f in self.spec.payload_fields}:
            raise FeatureError("原表映射须准确覆盖Schema全部字段")
        self.config, self.clock_ns = config, clock_ns or time.time_ns
        self.source_identity = config.identity + ":" + self.spec.fingerprint
        self.transport = DolphinDbReadSession(config.connection, session_factory=session_factory)
        self._types, self._observed, self._last_clock = None, {}, -1
        self._lock = RLock()

    def _table(self):
        return f'loadTable("{self.config.database}","{self.config.table}")'

    def inspect(self, *, sample_rows=3, sample_key=None):
        if type(sample_rows) is not int or not 0 <= sample_rows <= 10:
            raise FeatureError("Schema探针最多读取10条样例")
        frame = self.transport.run(f"schema({self._table()}).colDefs")
        if not hasattr(frame, "to_dict"):
            raise FeatureError("Schema须返回colDefs表")
        result = {"database": self.config.database, "table": self.config.table, "colDefs": frame.to_dict("records")}
        if sample_rows:
            condition = f" where {self.config.key_column}={literal(sample_key)}" if sample_key is not None else ""
            sample = self.transport.run(f"select top {sample_rows} * from {self._table()}{condition}")
            result["sample"] = sample.to_dict("records")
        return result

    def open(self):
        with self._lock:
            if self._types is not None:
                return
            self.transport.open()
            try:
                definitions = self.inspect(sample_rows=0)["colDefs"]
                self._types = {str(r["name"]): str(r["typeString"]).upper() for r in definitions}
                if len(self._types) != len(definitions) or self._types.get(self.config.key_column) not in {"STRING", "SYMBOL"}:
                    raise FeatureError("Schema列须唯一；业务键须为STRING/SYMBOL")
                numeric = {"CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE", "DECIMAL32", "DECIMAL64", "DECIMAL128"}
                integers = {"CHAR", "SHORT", "INT", "LONG"}
                for field in self.spec.payload_fields:
                    kind = self._types.get(self.config.payload_columns[field.name], "").split("(")[0]
                    allowed = integers if field.kind == "int" else numeric if field.kind in {"decimal", "float"} \
                        else {"BOOL"} if field.kind == "bool" else {"STRING", "SYMBOL"}
                    if kind not in allowed:
                        raise FeatureError(f"源字段{field.name}类型与声明Schema不兼容")
                if self.config.partition_date and self._types.get(self.config.partition_date) != "DATE":
                    raise FeatureError("日期分区须为DATE")
                for name in (self.config.revision_column, self.config.quality_column):
                    if name is not None and self._types.get(name) not in integers:
                        raise FeatureError("修订/质量列须为整数")
                self._timestamp(self.config.event_time)
                if self.config.publication_time is not None:
                    self._timestamp(self.config.publication_time)
            except Exception:
                self._types = None
                self.transport.close()
                raise

    def close(self):
        with self._lock:
            self._types = None
            self.transport.close()

    def _timestamp(self, mapping):
        kind = self._types.get(mapping.column)
        if mapping.encoding == "unix_ns":
            if kind not in {"LONG", "INT"}:
                raise FeatureError("unix_ns须为整数源列")
            return mapping.column
        if kind in {"SECOND", "TIME", "NANOTIME", "MINUTE"}:
            if mapping.date_column is None or self._types.get(mapping.date_column) != "DATE":
                raise FeatureError("日内时间须显式配置DATE拼接列")
            value = f"time({mapping.column})" if kind == "MINUTE" else mapping.column
            expression = f"concatDateTime({mapping.date_column},{value})"
        elif kind in {"DATETIME", "TIMESTAMP", "NANOTIMESTAMP", "DATEHOUR"}:
            expression = mapping.column
        else:
            raise FeatureError(f"时间列{mapping.column}的类型{kind}未受支持")
        expression = f"long(nanotimestamp({expression}))"
        return f"({expression}-28800000000000l)" if mapping.timezone == "Asia/Shanghai" else expression

    def _script(self, request):
        c = self.config
        if (request.dataset_id, request.schema_version) != (c.dataset_id, c.schema_version):
            raise FeatureError("请求与原表绑定的数据集不匹配")
        event = self._timestamp(c.event_time)
        offset = {"start": self.spec.frequency.period_ns, "end": 0, "inclusive-end": 1}[c.event_time.label]
        if offset:
            event = f"({event}+{offset}l)"
        published = self._timestamp(c.publication_time) if c.publication_time else "0l"
        conditions = [f'{c.key_column} in [{",".join(literal(key) for key in request.keys)}]',
            f"{event}>={request.start_ns}l", f"{event}<={request.end_ns}l"]
        if c.partition_date:
            zone = ZoneInfo(c.event_time.timezone)
            first = datetime.fromtimestamp(max(0, request.start_ns-offset)//1_000_000_000, zone).strftime("%Y.%m.%d")
            last = datetime.fromtimestamp(request.end_ns//1_000_000_000, zone).strftime("%Y.%m.%d")
            conditions += [f"{c.partition_date}>={first}", f"{c.partition_date}<={last}"]
        if c.availability == "source-time":
            conditions.append(f"{published}<={request.as_of_ns}l")
        fields = [f"{c.key_column} as key", f"{event} as event_ns", f"{published} as published_ns"]
        fields += [f"{c.payload_columns[f.name]} as p_{f.name}" for f in self.spec.payload_fields]
        if c.revision_column:
            fields.append(f"{c.revision_column} as revision_ns")
        if c.quality_column:
            fields.append(f"{c.quality_column} as quality_flags")
        order = list(dict.fromkeys(name for name in (c.event_time.date_column, c.event_time.column,
            c.key_column, c.publication_time.column if c.publication_time else None, c.revision_column) if name))
        return f"select top {request.max_rows+1} " + ",".join(fields) + f" from {self._table()} where " \
            + ",".join(conditions) + " order by " + ",".join(order)

    def read(self, request):
        with self._lock:
            if self._types is None:
                raise FeatureSourceError("须先open原表Provider并核验Schema")
            frame = self.transport.run(self._script(request))
            if len(frame) > request.max_rows:
                raise FeatureSourceError("原表查询超过上限，不能截断为完整窗口")
            if not frame.columns.is_unique:
                raise FeatureError("原表查询结果有重复列")
            import pandas as pd
            raw = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
            observed = self.clock_ns()
            if type(observed) is not int or observed < self._last_clock:
                raise FeatureError("原表观测时钟无效或回退")
            staged, records, within_batch = dict(self._observed), [], {}
            for position, row in enumerate(raw, 1):
                try:
                    key, event = row["key"], _integer(row["event_ns"], "event_ns")
                    if type(key) is not str or key not in request.keys:
                        raise FeatureError("原表业务键不属于请求范围或已丢失字符串格式")
                    if event > observed or not request.start_ns <= event <= request.end_ns:
                        raise FeatureError("原表业务时间尚未结束或不属于请求范围")
                    self.spec.frequency.close_ns(event)
                    payload = self.spec.payload_type(**{f.name: _payload_value(row["p_"+f.name], f)
                        for f in self.spec.payload_fields})
                    quality = _integer(row["quality_flags"], "quality_flags") if self.config.quality_column else 0
                    if self.config.availability == "source-time":
                        if row["published_ns"] is None:
                            raise FeatureError("source-time模式不允许空发布时间")
                        published = _integer(row["published_ns"], "发布时间")
                        if published > request.as_of_ns:
                            raise FeatureError("原表返回尚未发布的记录")
                        revision = _integer(row["revision_ns"], "revision") if self.config.revision_column else published
                        record = ExternalDataRecord(request.dataset_id, request.schema_version, key, payload,
                            event, published, event, revision, self.source_identity, quality)
                    else:
                        identity, content = (key, event), (payload, quality | 2)
                        if identity in within_batch and within_batch[identity] != content:
                            raise FeatureError("observed单批同一业务槽存在冲突快照")
                        within_batch[identity] = content
                        previous = staged.get(identity)
                        if previous is not None and (previous.payload, previous.quality_flags) == content:
                            record = previous
                        else:
                            if previous is not None and observed <= previous.available_ns:
                                raise FeatureError("内容更新须有严格递增的观测版本时间")
                            record = ExternalDataRecord(request.dataset_id, request.schema_version, key, payload,
                                event, observed, event, observed, self.source_identity, quality | 2)
                            staged[identity] = record
                    records.append(record)
                except (KeyError, FeatureError) as error:
                    raise FeatureError(f"原表记录{position}: {error}") from None
            cutoff = replace(request, as_of_ns=max(request.as_of_ns, observed)) if self.config.availability == "observed" else request
            batch = select_records(self.registry, records, cutoff, self.source_identity)
            self._observed, self._last_clock = staged, observed
            return batch
