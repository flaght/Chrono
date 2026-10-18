"""带Schema清单与内容摘要的特征文件；纯本地读写，保留全部修订。"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from bomber.framework.datahub.features import FeatureError, FeatureStore
from .base import FeatureProvider, FeatureSourceError, select_records
from .codec import FeatureColumnMapping, decode_record, encode_record, mapping_digest


def manifest_path(path):
    return Path(str(path) + ".manifest.json")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_features(path, registry, records):
    """不覆盖已有包；manifest作为最后发布标记，消费者拒绝半包。"""
    path = Path(path)
    if path.suffix.lower() not in {".csv", ".parquet"}:
        raise FeatureError("特征文件首期只支持CSV/Parquet")
    rows = tuple(records)
    if not rows:
        raise FeatureError("导出需要至少一条有效记录")
    identity = (rows[0].dataset_id, rows[0].schema_version)
    if any((row.dataset_id, row.schema_version) != identity for row in rows):
        raise FeatureError("单文件须为同一数据集Schema，多个业务键/修订可以共存")
    store = FeatureStore(registry)
    store.publish(rows)
    _, rows = store.capture()
    spec = registry.get(*identity)
    encoded = [encode_record(row) for row in sorted(rows, key=lambda row: (row.available_ns, row.identity))]
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = manifest_path(path)
    if path.exists() or sidecar.exists():
        raise FeatureError("导出目标已存在，不能覆盖历史版本")
    descriptor, temporary = tempfile.mkstemp(prefix=".feature-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    installed = False
    installed_manifest = False
    try:
        if path.suffix.lower() == ".csv":
            with open(temporary, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(encoded[0]))
                writer.writeheader()
                for row in encoded:
                    writer.writerow({key: str(value).lower() if type(value) is bool else value for key, value in row.items()})
                stream.flush()
                os.fsync(stream.fileno())
        else:
            import pandas as pd
            # received_ns有空值时保留nullable int64，避免纳秒被转为float。
            frame = pd.DataFrame(encoded)
            frame["received_ns"] = pd.array([row["received_ns"] for row in encoded], dtype="Int64")
            frame.to_parquet(temporary, index=False)
        manifest = {"format_version": 1, "dataset_id": identity[0], "schema_version": identity[1],
            "schema_fingerprint": spec.fingerprint, "rows": len(encoded), "sha256": _hash(temporary),
            "timestamp_unit": "unix_ns", "timestamp_label": "end", "publication_evidence": "record_available_ns",
            "columns": tuple(encoded[0])}
        os.link(temporary, path)
        installed = True
        with sidecar.open("x", encoding="utf-8") as stream:
            installed_manifest = True
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if installed:
            path.unlink(missing_ok=True)
        if installed_manifest:
            sidecar.unlink(missing_ok=True)
        raise
    finally:
        os.unlink(temporary)
    return manifest


class FileFeatureProvider(FeatureProvider):
    backend = "file"
    storage_kind = "file"

    def __init__(self, path, registry, *, mapping=None, require_manifest=True):
        super().__init__(registry)
        self.path = Path(path)
        self.mapping = mapping or FeatureColumnMapping()
        self.require_manifest = require_manifest
        self._records = None
        self.source_identity = None

    def open(self):
        if self._records is not None:
            return
        # 校验和解析同一份字节，防止文件在两次读取之间被替换。
        content = self.path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        manifest = None
        if self.require_manifest:
            manifest = json.loads(manifest_path(self.path).read_text(encoding="utf-8"))
            if manifest.get("format_version") != 1 or manifest.get("timestamp_unit") != "unix_ns" \
                or manifest.get("timestamp_label") != self.mapping.timestamp_label or manifest.get("sha256") != digest:
                raise FeatureError("文件manifest格式/时间规则/内容校验失败")
            spec = self.registry.get(manifest["dataset_id"], manifest["schema_version"])
            if manifest.get("schema_fingerprint") != spec.fingerprint:
                raise FeatureError("文件Schema或计算版本与注册表不兼容")
        if self.path.suffix.lower() == ".csv":
            with io.StringIO(content.decode("utf-8-sig"), newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise FeatureError("CSV字段为空或重复")
                raw = list(reader)
        elif self.path.suffix.lower() == ".parquet":
            import pandas as pd
            frame = pd.read_parquet(io.BytesIO(content))
            if not frame.columns.is_unique:
                raise FeatureError("Parquet字段重复")
            raw = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        else:
            raise FeatureError("特征文件只支持CSV/Parquet")
        rows = []
        for position, row in enumerate(raw, 2):
            try:
                if None in row:
                    raise FeatureError("CSV记录字段数量超过表头")
                rows.append(decode_record(row, self.registry, self.mapping))
            except FeatureError as error:
                raise FeatureError(f"记录{position}: {error}") from None
        if manifest and (len(rows) != manifest["rows"] or any(
            (row.dataset_id, row.schema_version) != (manifest["dataset_id"], manifest["schema_version"]) for row in rows
        )):
            raise FeatureError("文件行数/数据集身份与manifest不一致")
        store = FeatureStore(self.registry)
        store.publish(rows)
        _, records = store.capture()
        self._records = records
        self.source_identity = f"file:{self.path.resolve()}:{digest}:{mapping_digest(self.mapping)}"

    def close(self):
        self._records = None

    def read(self, request):
        if self._records is None:
            raise FeatureSourceError("须先open特征文件")
        return select_records(self.registry, self._records, request, self.source_identity)
