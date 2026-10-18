"""通用特征轮询缓存；因子和行情都在同一DataHub消费，不依赖原生Bar。"""
import math
from datetime import datetime
from threading import RLock
import time
from zoneinfo import ZoneInfo

from bomber.framework.datahub import DataHub
from bomber.framework.datahub.features import FeatureError, FeatureHub
from .base import FeatureRequest
from .service import FeatureRefreshService


class FeatureQueryCache:
    def __init__(self, provider, *, dataset_id, schema_version, keys, start_ns, end_ns,
                 poll_seconds=.2, clock_ns=None, source_id="feature-query", hub=None, max_rows=10_000):
        if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 30:
            raise FeatureError("特征查询间隔须在(0,30]秒")
        self.provider, self.source_id = provider, source_id
        self.dataset_id, self.schema_version = dataset_id, schema_version
        self.spec = provider.registry.get(dataset_id, schema_version)
        self.keys, self.start_ns, self.end_ns = tuple(keys), start_ns, end_ns
        FeatureRequest(dataset_id, schema_version, self.keys, start_ns, end_ns, end_ns, max_rows=max_rows)
        self.max_rows = max_rows
        self.clock_ns = clock_ns or time.time_ns
        self.poll_ns = int(poll_seconds * 1e9)
        self.hub = hub if hub is not None else DataHub(features=FeatureHub(provider.registry))
        self.service = FeatureRefreshService(self.hub, {(dataset_id, schema_version): provider})
        self.connected, self.failure, self.last_query_ns = False, None, None
        self._lock = RLock()

    def open(self):
        with self._lock:
            if self.connected:
                return
            try:
                self.provider.open()
                self.connected = True
                self.poll(force=True)
            except Exception:
                self.connected = False
                self.provider.close()
                raise

    def close(self):
        with self._lock:
            self.connected = False
            self.provider.close()

    @property
    def manifest(self):
        with self._lock:
            generation, rows = self.hub.features.store.capture()
            count = sum((r.dataset_id, r.schema_version) == (self.dataset_id, self.schema_version)
                and r.key in self.keys and self.start_ns <= r.ts_event <= self.end_ns for r in rows)
            selected = [r for r in rows if (r.dataset_id, r.schema_version) == (self.dataset_id, self.schema_version)
                and r.key in self.keys and self.start_ns <= r.ts_event <= self.end_ns]
            latest = max(selected, key=lambda r: (r.ts_event, r.revision)) if selected else None
            return {"connected": self.connected, "failure": self.failure, "last_query_ns": self.last_query_ns,
                "cached_records": count, "generation": generation, "dataset_id": self.dataset_id,
                "cached_slots": len({(r.key, r.ts_event) for r in selected}),
                "latest_event_ns": None if latest is None else latest.ts_event,
                "latest_event_time": None if latest is None else datetime.fromtimestamp(
                    latest.ts_event / 1e9, ZoneInfo(self.spec.frequency.timezone)).isoformat(),
                "latest_available_ns": None if latest is None else latest.available_ns,
                "latest_received_ns": None if latest is None else latest.received_ns,
                "schema_version": self.schema_version, "keys": self.keys, "schema_fingerprint": self.spec.fingerprint,
                "backend": getattr(self.provider, "backend", self.provider.storage_kind),
                "source_id": self.source_id}

    def poll(self, *, force=False):
        with self._lock:
            if not self.connected:
                raise RuntimeError("特征查询缓存未连接")
            now = self.clock_ns()
            try:
                if self.last_query_ns is not None and now < self.last_query_ns:
                    raise FeatureError("特征查询时钟回退")
                if force or self.last_query_ns is None or now-self.last_query_ns >= self.poll_ns:
                    frequency = self.spec.frequency
                    end = min(self.end_ns, (now-frequency.anchor_ns)//frequency.period_ns
                        * frequency.period_ns + frequency.anchor_ns)
                    if end < self.start_ns:
                        raise FeatureError("特征读取业务区间尚未开始")
                    request = FeatureRequest(self.dataset_id, self.schema_version, self.keys,
                        self.start_ns, end, now, max_rows=self.max_rows)
                    self.service.refresh((request,), clock_ns=self.clock_ns)
                    self.last_query_ns = self.clock_ns()
                return self.hub.features.store.generation
            except Exception as error:
                self.failure = str(error)
                raise
