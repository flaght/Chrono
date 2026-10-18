"""显式配置的DolphinDB/MongoDB特征适配；只有open/read联网。"""
from dataclasses import dataclass, field
import json
import re
from threading import RLock

from bomber.framework.datahub.features import FeatureError
from .base import FeatureProvider, FeatureSourceError, select_records
from .codec import ENVELOPE_COLUMNS, FeatureColumnMapping, decode_record, mapping_digest


def identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise FeatureError("数据库表/列须为普通标识符，不接受SQL片段")
    return value


@dataclass(frozen=True)
class DatasetTable:
    dataset_id: str
    schema_version: int
    table: str
    mapping: FeatureColumnMapping = field(default_factory=FeatureColumnMapping)

    def __post_init__(self):
        identifier(self.table)
        for column in self.mapping.columns.values():
            identifier(column)


@dataclass(frozen=True)
class DolphinDbFeatureConfig:
    host: str
    port: int
    username: str = field(repr=False)
    password: str = field(repr=False)
    database: str
    tables: tuple[DatasetTable, ...]
    timeout_seconds: int = 15
    enable_ssl: bool = False

    def __post_init__(self):
        if not self.host or not self.username or not self.password or type(self.port) is not int or not 1 <= self.port <= 65535:
            raise FeatureError("DolphinDB连接配置不完整")
        if re.fullmatch(r"dfs://[A-Za-z0-9_/-]+", self.database) is None:
            raise FeatureError("DolphinDB须显式提供安全的DFS数据库路径")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 120:
            raise FeatureError("查询超时须在1至120秒")
        object.__setattr__(self, "tables", tuple(self.tables))


@dataclass(frozen=True)
class MongoDbFeatureConfig:
    uri: str = field(repr=False)
    database: str
    tables: tuple[DatasetTable, ...]
    source_id: str
    timeout_seconds: int = 15

    def __post_init__(self):
        if not self.uri.startswith(("mongodb://", "mongodb+srv://")) or not self.source_id:
            raise FeatureError("MongoDB须提供URI和独立来源标识")
        identifier(self.database)
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 120:
            raise FeatureError("查询超时须在1至120秒")
        object.__setattr__(self, "tables", tuple(self.tables))


class _DatabaseFeatures(FeatureProvider):
    def __init__(self, config, registry):
        super().__init__(registry)
        self.config = config
        self._tables = {(item.dataset_id, item.schema_version): item for item in config.tables}
        if not self._tables or len(self._tables) != len(config.tables):
            raise FeatureError("数据库数据集映射不能为空或重复")
        self._lock = RLock()
        for item in config.tables:
            spec = registry.get(item.dataset_id, item.schema_version)
            # 数据库需要普通实际列名；payload.xxx应显式映射，如close/score。
            for name in (*ENVELOPE_COLUMNS, *("payload." + f.name for f in spec.payload_fields)):
                identifier(item.mapping.column(name))

    def binding(self, request):
        try:
            return self._tables[(request.dataset_id, request.schema_version)]
        except KeyError:
            raise FeatureError("数据库未配置所需数据集Schema映射") from None

    def decode(self, raw, request, binding, identity):
        rows = []
        for index, row in enumerate(raw, 1):
            try:
                decoded = decode_record(row, self.registry, binding.mapping)
            except FeatureError as error:
                raise FeatureError(f"数据库记录{index}: {error}") from None
            if (decoded.dataset_id, decoded.schema_version) != (request.dataset_id, request.schema_version):
                raise FeatureError("数据库返回其他数据集/Schema")
            rows.append(decoded)
        return select_records(self.registry, rows, request, identity)


class DolphinDbFeatureProvider(_DatabaseFeatures):
    backend = "dolphindb"
    def __init__(self, config, registry, *, session_factory=None):
        if not isinstance(config, DolphinDbFeatureConfig):
            raise FeatureError("DolphinDB特征须使用独立配置")
        super().__init__(config, registry)
        self._factory = session_factory
        self._session = None

    def open(self):
        with self._lock:
            if self._session is not None:
                return
            session = None
            try:
                factory = self._factory
                if factory is None:
                    import dolphindb
                    factory = dolphindb.session
                session = factory(enableSSL=self.config.enable_ssl)
                if session.connect(self.config.host, self.config.port, self.config.username, self.config.password,
                    reconnect=False, readTimeout=self.config.timeout_seconds,
                    writeTimeout=self.config.timeout_seconds) is not True:
                    raise FeatureSourceError("连接未确认成功")
                self._session = session
            except Exception:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass
                raise FeatureSourceError("DolphinDB特征连接失败；检查网络/SDK/配置") from None

    def close(self):
        with self._lock:
            session, self._session = self._session, None
            if session is not None:
                try:
                    session.close()
                except Exception:
                    raise FeatureSourceError("DolphinDB特征会话关闭失败") from None

    def _script(self, request, binding):
        def literal(value):
            if any(ord(char) < 32 or char in {'"', "\\"} for char in value):
                raise FeatureError("DolphinDB业务标识不能包含控制字符、引号或反斜线")
            return json.dumps(value, ensure_ascii=False)
        column = binding.mapping.column
        # 输入源必须先提供整数纳秒标准列；非标准时间列通过上游视图规范化。
        event_offset = {"end": 0, "start": self.registry.get(request.dataset_id,
            request.schema_version).frequency.period_ns, "inclusive-end": 1}[binding.mapping.timestamp_label]
        conditions = [f'{column("dataset_id")} = {literal(request.dataset_id)}',
            f'{column("schema_version")} = {request.schema_version}',
            f'{column("key")} in [{",".join(literal(key) for key in request.keys)}]',
            f'{column("ts_event")} >= {request.start_ns - event_offset}',
            f'{column("ts_event")} <= {request.end_ns - event_offset}',
            f'{column("available_ns")} <= {request.as_of_ns}']
        return (f'select top {request.max_rows + 1} * from loadTable("{self.config.database}", '
            f'"{binding.table}") where ' + ", ".join(conditions))

    def read(self, request):
        binding = self.binding(request)
        script = self._script(request, binding)
        with self._lock:
            if self._session is None:
                raise FeatureSourceError("须先open DolphinDB特征来源")
            try:
                frame = self._session.run(script)
            except Exception:
                raise FeatureSourceError("DolphinDB特征查询失败；不解释为空数据") from None
        if len(frame) > request.max_rows:
            raise FeatureSourceError("DolphinDB特征查询超限，不返回截断数据")
        if not frame.columns.is_unique:
            raise FeatureError("数据库返回重复列")
        import pandas as pd
        raw = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        identity = f"dolphindb:{self.config.host}:{self.config.port}:{self.config.database}/{binding.table}:{mapping_digest(binding.mapping)}"
        return self.decode(raw, request, binding, identity)


class MongoDbFeatureProvider(_DatabaseFeatures):
    backend = "mongodb"
    def __init__(self, config, registry, *, client_factory=None):
        if not isinstance(config, MongoDbFeatureConfig):
            raise FeatureError("MongoDB特征须使用独立配置")
        super().__init__(config, registry)
        self._factory = client_factory
        self._client = None

    def open(self):
        with self._lock:
            if self._client is not None:
                return
            client = None
            try:
                factory = self._factory
                if factory is None:
                    from pymongo import MongoClient
                    factory = MongoClient
                milliseconds = self.config.timeout_seconds * 1000
                client = factory(self.config.uri, serverSelectionTimeoutMS=milliseconds,
                    connectTimeoutMS=milliseconds, socketTimeoutMS=milliseconds)
                client.admin.command("ping")
                self._client = client
            except Exception:
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
                raise FeatureSourceError("MongoDB特征连接失败；检查网络/SDK/配置") from None

    def close(self):
        with self._lock:
            client, self._client = self._client, None
            if client is not None:
                try:
                    client.close()
                except Exception:
                    raise FeatureSourceError("MongoDB特征会话关闭失败") from None

    def read(self, request):
        binding = self.binding(request)
        column = binding.mapping.column
        offset = {"end": 0, "start": self.registry.get(request.dataset_id,
            request.schema_version).frequency.period_ns, "inclusive-end": 1}[binding.mapping.timestamp_label]
        query = {column("dataset_id"): request.dataset_id, column("schema_version"): request.schema_version,
            column("key"): {"$in": list(request.keys)}, column("ts_event"): {
                "$gte": request.start_ns - offset, "$lte": request.end_ns - offset},
            column("available_ns"): {"$lte": request.as_of_ns}}
        with self._lock:
            if self._client is None:
                raise FeatureSourceError("须先open MongoDB特征来源")
            cursor = None
            try:
                collection = self._client[self.config.database][binding.table]
                cursor = collection.find(query, max_time_ms=self.config.timeout_seconds * 1000).limit(request.max_rows + 1)
                raw = list(cursor)
            except Exception:
                raise FeatureSourceError("MongoDB特征查询失败；不解释为空数据") from None
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        raise FeatureSourceError("MongoDB特征查询游标关闭失败") from None
        if len(raw) > request.max_rows:
            raise FeatureSourceError("MongoDB特征查询超限，不返回截断数据")
        identity = f"mongodb:{self.config.source_id}:{self.config.database}/{binding.table}:{mapping_digest(binding.mapping)}"
        return self.decode(raw, request, binding, identity)
