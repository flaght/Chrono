"""控制线程发布参考快照；行情线程只读缓存，刷新失败不沿用旧版本。"""

from copy import deepcopy
from threading import RLock
import time


class LiveReferenceCache:
    def __init__(self, *, clock_ns=None):
        self.clock_ns = clock_ns or (lambda: time.time_ns())
        self.publication_lock = RLock()
        self._lock = RLock()
        self._value = None
        self._failure = None
        self._clock_high_water_ns = None

    def refresh(self, loader, validate=None):
        try:
            # 查询不持提交锁；快照、条款、manifest作为一个版本发布。
            value = loader()
            manifest = deepcopy(value[2])
            if not manifest.get("ready", True):
                raise RuntimeError(manifest.get("failure") or "参考资料未就绪")
            manifest.setdefault("observed_at_ns", self.clock_ns())
            manifest.setdefault("freshness_policy", {"max_observation_age_seconds": 30})
            with self.publication_lock, self._lock:
                self._value = (value[0], value[1], manifest)
                self._failure = None
                if validate:
                    validate()
        except Exception as error:
            with self._lock:
                self._failure = str(error)
            raise

    @property
    def value(self):
        with self._lock:
            return self._value

    @property
    def manifest(self):
        with self._lock:
            result = deepcopy(self._value[2]) if self._value else {}
            failure = self._failure
            if failure is None and self._value:
                now = self.clock_ns()
                age = now - result["observed_at_ns"]
                limit = result["freshness_policy"]["max_observation_age_seconds"]
                if self._clock_high_water_ns is not None and now < self._clock_high_water_ns:
                    failure = self._failure = "参考缓存时钟回退"
                elif not 0 <= age <= limit * 1_000_000_000:
                    failure = "参考缓存观测过期或时钟回退"
                self._clock_high_water_ns = max(now, self._clock_high_water_ns or now)
            return {**result, "ready": failure is None and self._value is not None,
                    "failure": failure}

    def snapshot(self, as_of_ns):
        with self._lock:
            manifest = self.manifest
            if not manifest["ready"]:
                raise RuntimeError(manifest["failure"])
            assignment = self._value[0]
            if type(as_of_ns) is not int or as_of_ns < max(assignment.effective_ns, assignment.available_ns):
                raise RuntimeError("当前参考版本在Bar时刻尚不可见")
            return assignment
