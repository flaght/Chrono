"""有时间证据的参考资料导出与纯本地读取；不自动连接数据库。"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from .base import ReferenceBatch, ReferenceDataSource, ReferenceDataset, ReferenceQuery, ReferenceSourceError
from .normalize import normalize_frame
from .policy import DataSourcePurpose, validate_source


def _encode(value):
    if isinstance(value, datetime) and hasattr(value, "nanosecond"):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if type(value) is date:
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("导出不接受非有限Decimal")
        return {"type": "decimal", "value": str(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    # pandas/numpy标量只能转成上述受控类型；不接受任意对象字符串化。
    if hasattr(value, "item"):
        return _encode(value.item())
    raise TypeError(f"参考导出不支持字段类型: {type(value).__name__}")


def _decode(value):
    if not isinstance(value, dict):
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError("导出字段须为标量")
    if set(value) != {"type", "value"}:
        raise ValueError("无效的导出字段外壳")
    types = {"date": date.fromisoformat, "datetime": datetime.fromisoformat, "decimal": Decimal}
    if value["type"] == "timestamp":
        import pandas as pd
        return pd.Timestamp(value["value"])
    if value["type"] not in types:
        raise ValueError("未知导出字段类型")
    return types[value["type"]](value["value"])


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _available(row):
    try:
        value = Decimal(str(row.get("available_ns")))
    except InvalidOperation:
        raise ValueError("导出记录须有真实available_ns，缺失时只能显式选择observed快照") from None
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise ValueError("导出记录须有非负整数available_ns；不能将来源日或导出时间当作历史发布时间")
    return int(value)


def _business_key(dataset, row):
    if dataset in {ReferenceDataset.FUTURES_BASIC, ReferenceDataset.OPTIONS_BASIC}:
        return (row["symbol"],)
    if dataset is ReferenceDataset.ADJUSTMENT_FACTORS:
        return (row["trade_date"], row["code"], row.get("role", "main"))
    return (row["date"], row["code"])


def export_reference_bundle(source, queries, destination, *, visibility="explicit", clock_ns=None):
    """导出声明范围；两轮一致不声称跨表事务性或完整历史版本。

    explicit要求每条源记录具有真实available_ns；observed只允许从稳定
    导出观测时刻起使用，不能拿当前快照回测更早事件。输出不覆盖旧文件。
    """
    if visibility not in {"explicit", "observed"}:
        raise ValueError("导出visibility须为explicit或observed")
    requests = {ReferenceDataset(key): value for key, value in queries.items()}
    if not requests or any(not isinstance(query, ReferenceQuery) for query in requests.values()):
        raise ValueError("导出须声明非空数据集与ReferenceQuery范围")
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("导出文件已存在；保留旧版本，选择新文件")
    clock = clock_ns or time.time_ns
    started = clock()
    first = {dataset: source.read(dataset, query) for dataset, query in requests.items()}
    second = {dataset: source.read(dataset, query) for dataset, query in requests.items()}
    if any(first[dataset].fingerprint != batch.fingerprint for dataset, batch in second.items()):
        raise ReferenceSourceError("导出期间资料变化；不会生成部分导出文件")
    observed = clock()
    if type(started) is not int or type(observed) is not int or not 0 <= started <= observed:
        raise ValueError("导出时钟须为非负整数且不能回退")
    batches, audit = {}, {}
    for dataset, batch in second.items():
        if not batch.rows:
            raise ReferenceSourceError(f"{dataset.value}导出范围没有资料")
        rows = []
        for row in batch.rows:
            values = dict(row)
            if visibility == "observed":
                values["available_ns"] = max(observed, _available(values) if "available_ns" in values else 0)
            else:
                _available(values)
            rows.append(values)
        columns = tuple(dict.fromkeys((*batch.columns, "available_ns")))
        exported = ReferenceBatch(dataset, tuple(rows), columns, batch.source)
        times = [_available(row) for row in exported.rows]
        # 相同业务键/可用时间/修订的冲突不能用文件行序解决。
        _visible_rows(exported, max(times))
        batches[dataset.value] = {"source": batch.source, "columns": list(columns),
            "rows": [{key: _encode(value) for key, value in row.items()} for row in exported.rows],
            "sha256": exported.fingerprint}
        audit[dataset.value] = {"query": {key: _encode(value) if not isinstance(value, tuple) else list(value)
            for key, value in asdict(requests[dataset]).items()}, "rows": len(rows),
            "source": batch.source, "source_content_sha256": batch.fingerprint,
            "exported_content_sha256": exported.fingerprint,
            "available_ns_min": min(times), "available_ns_max": max(times),
            "source_versions": sorted({str(row["source_version"]) for row in rows if row.get("source_version") is not None}),
            "revision_evidence": "source_fields_if_present_otherwise_content_hash"}
    config = getattr(source, "config", None)
    identity = ({"host": config.host, "port": config.port, "database": config.database}
        if config is not None and all(hasattr(config, name) for name in ("host", "port", "database")) else None)
    payload = {"manifest": {"schema_version": 1, "purpose": DataSourcePurpose.EXPORT.value,
        "started_ns": started, "observed_ns": observed, "visibility": visibility,
        "publication_evidence": "source_available_ns" if visibility == "explicit" else "observation_lower_bound_only",
        "history_evidence": "only_exported_versions_not_complete_upstream_history",
        "consistency_evidence": "two_equal_reads_not_transaction_snapshot",
        "source_identity": identity, "datasets": audit}, "batches": batches}
    envelope = {"payload": payload, "sha256": _digest(payload)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".reference-export-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_json(envelope) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # 原子发布且不覆盖并发导出的同名文件。
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    return envelope


def _visible_rows(batch, as_of_ns):
    if type(as_of_ns) is not int or as_of_ns < 0:
        raise ValueError("as_of_ns须为非负整数")
    selected, seen = {}, {}
    for row in batch.rows:
        available = _available(row)
        if available > as_of_ns:
            continue
        revision = row.get("revision", 0)
        if type(revision) is not int or revision < 0:
            raise ValueError("revision须为非负整数")
        key, version = _business_key(batch.dataset, row), (available, revision)
        identity = (key, version)
        if identity in seen and dict(seen[identity]) != dict(row):
            raise ValueError("相同业务键、可用时间及修订存在冲突")
        seen[identity] = row
        previous = selected.get(key)
        if previous is None or version > previous[0]:
            selected[key] = (version, row)
    return tuple(value[1] for value in selected.values())


class ExportedReferenceSource(ReferenceDataSource):
    """只读本地导出，不持有数据库配置或连接；查询显式带可见截止点。"""

    def __init__(self, path, *, as_of_ns, purpose=DataSourcePurpose.OFFLINE_REFERENCE):
        validate_source("file", purpose)
        if DataSourcePurpose(purpose) is not DataSourcePurpose.OFFLINE_REFERENCE:
            raise ValueError("参考导出仅用于offline_reference消费")
        if type(as_of_ns) is not int or as_of_ns < 0:
            raise ValueError("as_of_ns须为非负整数")
        self.path = Path(path).expanduser().resolve()
        self.as_of_ns = as_of_ns
        self._batches = None
        self._manifest = None

    def open(self):
        if self._batches is not None:
            return
        def reject_constant(value):
            raise ValueError(f"非有限JSON常量: {value}")
        data = json.loads(self.path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        if set(data) != {"payload", "sha256"} or _digest(data["payload"]) != data["sha256"]:
            raise ValueError("参考导出文件校验失败")
        payload = data["payload"]
        manifest = payload["manifest"]
        if manifest["schema_version"] != 1 or manifest["purpose"] != DataSourcePurpose.EXPORT.value:
            raise ValueError("不支持的参考导出Schema或用途")
        if set(payload["batches"]) != set(manifest["datasets"]):
            raise ValueError("参考导出数据集与清单不一致")
        batches = {}
        for name, item in payload["batches"].items():
            batch = ReferenceBatch(ReferenceDataset(name),
                tuple({key: _decode(value) for key, value in row.items()} for row in item["rows"]),
                tuple(item["columns"]), item["source"])
            audit = manifest["datasets"][name]
            if (batch.fingerprint != item["sha256"] or batch.fingerprint != audit["exported_content_sha256"]
                    or len(batch.rows) != audit["rows"]):
                raise ValueError("参考导出记录与清单校验失败")
            if any(set(row) != set(batch.columns) for row in batch.rows):
                raise ValueError("参考导出行与Schema列不一致")
            _visible_rows(batch, max(_available(row) for row in batch.rows))
            batches[batch.dataset] = batch
        self._batches, self._manifest = batches, manifest

    @property
    def manifest(self):
        if self._batches is None:
            raise ReferenceSourceError("须先open本地导出")
        return json.loads(_json(self._manifest))

    def close(self):
        self._batches = self._manifest = None

    def read(self, dataset, query):
        return self.read_as_of(dataset, query, self.as_of_ns)

    def read_as_of(self, dataset, query, as_of_ns):
        if self._batches is None:
            raise ReferenceSourceError("须先open本地导出")
        if not isinstance(query, ReferenceQuery):
            raise TypeError("查询需要ReferenceQuery")
        dataset = ReferenceDataset(dataset)
        if dataset not in self._batches:
            raise ValueError(f"导出未包含{dataset.value}")
        import pandas as pd
        batch = self._batches[dataset]
        rows = _visible_rows(batch, as_of_ns)
        return normalize_frame(pd.DataFrame([dict(row) for row in rows], columns=batch.columns),
            dataset, query, source=batch.source)
