"""标准特征Provider及可重放增量游标；构造和导入不联网。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json

from bomber.framework.datahub.features.contracts import FeatureError, integer


class FeatureSourceError(RuntimeError):
    """连接/查询错误，与数据为空区分；不包含连接凭据。"""


@dataclass(frozen=True)
class FeatureRequest:
    dataset_id: str
    schema_version: int
    keys: tuple[str, ...]
    start_ns: int
    end_ns: int
    as_of_ns: int
    max_rows: int = 100_000

    def __post_init__(self):
        integer(self.schema_version, "schema_version", 1)
        for name in ("start_ns", "end_ns", "as_of_ns"):
            integer(getattr(self, name), name)
        integer(self.max_rows, "max_rows", 1)
        if self.max_rows > 1_000_000:
            raise FeatureError("单次读取上限不能超过1000000行")
        if not self.dataset_id or not self.keys or len(set(self.keys)) != len(self.keys) or any(
            type(key) is not str or not key for key in self.keys
        ) or self.start_ns > self.end_ns or self.end_ns > self.as_of_ns:
            raise FeatureError("请求须明确数据集、唯一键、业务区间及可见截止点")
        object.__setattr__(self, "keys", tuple(self.keys))

    @property
    def scope(self):
        # 截止点可推进；其余范围改变必须重新初始化游标。
        value = (self.dataset_id, self.schema_version, sorted(self.keys), self.start_ns)
        return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class FeatureCursor:
    source_identity: str
    scope: str
    as_of_ns: int
    end_ns: int
    seen: tuple[tuple, ...]


@dataclass(frozen=True)
class FeatureBatch:
    records: tuple
    source_identity: str
    schema_fingerprint: str
    cursor: FeatureCursor | None = None


class FeatureProvider(ABC):
    storage_kind = "database"

    def __init__(self, registry):
        self.registry = registry

    @abstractmethod
    def open(self): ...

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def read(self, request: FeatureRequest) -> FeatureBatch: ...

    def read_updates(self, request, cursor=None):
        """有界范围重读，按版本身份去重；发现旧业务槽新修订/迟到写入。

        不按业务时间推进游标，不假设发布时间就是数据库提交时间。
        游标记录内容指纹；范围太大失败，不返回截断成功。不宣称跨表事务快照。
        """
        batch = self.read(request)
        if cursor is not None and (cursor.source_identity != batch.source_identity
            or cursor.scope != request.scope or request.as_of_ns < cursor.as_of_ns
            or request.end_ns < cursor.end_ns):
            raise FeatureError("更新游标来源/范围改变或截止点回退")
        from .codec import record_digest
        seen = dict(cursor.seen) if cursor else {}
        changed = []
        for row in batch.records:
            digest = record_digest(row)
            old = seen.get(row.identity)
            if old is not None and old != digest:
                raise FeatureError("上游原地覆盖相同修订，不能丢弃为重复")
            if old is None:
                changed.append(row)
                seen[row.identity] = digest
        updated = FeatureCursor(batch.source_identity, request.scope, request.as_of_ns,
            request.end_ns, tuple(sorted(seen.items())))
        return FeatureBatch(tuple(changed), batch.source_identity, batch.schema_fingerprint, updated)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


def select_records(registry, records, request, source_identity):
    from bomber.framework.datahub.features import FeatureStore
    spec = registry.get(request.dataset_id, request.schema_version)
    selected = tuple(row for row in records if
        (row.dataset_id, row.schema_version) == (request.dataset_id, request.schema_version)
        and row.key in request.keys and request.start_ns <= row.ts_event <= request.end_ns
        and row.available_ns <= request.as_of_ns)
    if len(selected) > request.max_rows:
        raise FeatureSourceError("特征查询超过行数上限；缩小范围，不返回截断数据")
    store = FeatureStore(registry)
    store.publish(selected)
    _, rows = store.capture()
    return FeatureBatch(tuple(sorted(rows, key=lambda row: (row.available_ns, row.identity))),
        source_identity, spec.fingerprint)
