"""保留全部修订和接收证据；整批校验后原子发布。"""
from dataclasses import replace
from threading import RLock

from .contracts import ExternalDataRecord, FeatureError, integer


class FeatureStore:
    def __init__(self, registry):
        self.registry = registry
        self._rows = {}
        self._health = {}
        self._lock = RLock()
        self.generation = 0

    def publish(self, records, *, health_updates=()):
        with self._lock:
            staged = dict(self._rows)
            grouped = {}
            for existing in staged.values():
                grouped.setdefault(existing.identity[:-1], []).append(existing)
            for row in records:
                if not isinstance(row, ExternalDataRecord):
                    raise FeatureError("存储只接收ExternalDataRecord")
                spec = self.registry.get(row.dataset_id, row.schema_version)
                spec.validate_payload(row.payload)
                spec.frequency.close_ns(row.ts_event)
                old = staged.get(row.identity)
                if old is not None:
                    if replace(old, received_ns=None) != replace(row, received_ns=None):
                        raise FeatureError("相同业务槽与修订存在冲突")
                    continue  # 重读不改首次接收时间，不刷新源数据年龄。
                siblings = grouped.setdefault(row.identity[:-1], [])
                if any((item.revision < row.revision and item.available_ns > row.available_ns)
                       or (item.revision > row.revision and item.available_ns < row.available_ns)
                       for item in siblings):
                    raise FeatureError("修订序号与发布时间次序矛盾")
                staged[row.identity] = row
                siblings.append(row)
            health = {key: list(events) for key, events in self._health.items()}
            for identity, observed_ns, ready_until_ns, code in health_updates:
                if len(identity) != 3 or not identity[2]:
                    raise FeatureError("来源健康须明确数据集、Schema及业务键")
                self.registry.get(identity[0], identity[1])
                integer(observed_ns, "来源观测时间")
                integer(ready_until_ns, "来源有效截止时间")
                if ready_until_ns < observed_ns or code is not None and (type(code) is not str or not code):
                    raise FeatureError("来源健康期限或错误码无效")
                events = health.setdefault(identity, [])
                if events and observed_ns < events[-1][0]:
                    raise FeatureError("来源观测时钟回退")
                events.append((observed_ns, ready_until_ns, code))
            if staged != self._rows or health != self._health:
                self._rows = staged
                self._health = health
                self.generation += 1
            return self.generation

    def capture(self):
        with self._lock:
            return self.generation, tuple(self._rows.values())

    def capture_with_health(self):
        with self._lock:
            return self.generation, tuple(self._rows.values()), {
                key: tuple(events) for key, events in self._health.items()}
