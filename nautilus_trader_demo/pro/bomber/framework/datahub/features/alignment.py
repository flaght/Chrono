"""本地与在线共用可见性、质量和完整区间规则；查询只读缓存。"""
from dataclasses import dataclass
from types import MappingProxyType

from .contracts import Alignment, FeatureError, FeatureUnavailable
from .store import FeatureStore


@dataclass(frozen=True)
class FeatureSnapshot:
    query: object
    generation: int
    values: object
    schema_fingerprints: object

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "schema_fingerprints", MappingProxyType(dict(self.schema_fingerprints)))

    def latest(self, alias):
        return self.values[alias][-1]


class FeatureHub:
    def __init__(self, registry, store=None):
        self.registry = registry
        self.store = store or FeatureStore(registry)
        if self.store.registry is not registry:
            raise FeatureError("Hub与Store须共享注册表")

    def snapshot(self, requirements, query):
        requirements = tuple(requirements)
        if not requirements or len({r.alias for r in requirements}) != len(requirements):
            raise FeatureError("需求不能为空或含重复alias")
        generation, captured, health = self.store.capture_with_health()
        values, fingerprints = {}, {}
        for requirement in requirements:
            spec = self.registry.get(requirement.dataset_id, requirement.schema_version)
            mode = requirement.alignment or spec.default_alignment
            if mode is not Alignment.WINDOW and requirement.window_ns is not None:
                raise FeatureError("window_ns仅适用于WINDOW，不能被其他对齐方式忽略")
            source_events = health.get((requirement.dataset_id, requirement.schema_version, requirement.key))
            if source_events is not None:
                observed = [item for item in source_events if item[0] <= query.as_of_ns]
                if not observed:
                    raise FeatureUnavailable("NOT_READY", f"{requirement.alias}尚未有来源就绪证据")
                _, until, code = observed[-1]
                if code or query.as_of_ns > until:
                    raise FeatureUnavailable(code or "REFRESH_STALE", f"{requirement.alias}来源未就绪/刷新过期")
            visible = {}
            for row in captured:
                if (row.dataset_id, row.schema_version, row.key) != (
                    requirement.dataset_id, requirement.schema_version, requirement.key
                ) or row.ts_event > query.event_ns or row.visible_ns > query.as_of_ns:
                    continue
                old = visible.get(row.ts_event)
                if old is None or row.revision > old.revision:
                    visible[row.ts_event] = row
            if mode is Alignment.WINDOW:
                if requirement.window_ns is None or requirement.window_ns > query.event_ns:
                    raise FeatureError("WINDOW须声明有效的window_ns")
                spec.frequency.close_ns(query.event_ns)
                spec.frequency.close_ns(query.event_ns - requirement.window_ns)
                ends = spec.frequency.expected_ends(query.event_ns - requirement.window_ns, query.event_ns)
            elif mode is Alignment.EXACT:
                spec.frequency.close_ns(query.event_ns)
                ends = (query.event_ns,)
            else:
                ends = (max(visible),) if visible else (query.event_ns,)
            missing = tuple(end for end in ends if end not in visible)
            if missing:
                raise FeatureUnavailable("MISSING", f"{requirement.alias}缺少可见业务槽: {missing}")
            rows = tuple(visible[end] for end in ends)
            if any(row.quality_flags & spec.rejected_quality_flags for row in rows):
                raise FeatureUnavailable("QUALITY", f"{requirement.alias}最新可见版本质量不合格")
            age = requirement.max_source_age_ns
            if age is None:
                age = spec.max_source_age_ns
            # 窗口的历史成员本来就旧；年龄限制检查末槽，覆盖检查已经逐槽执行。
            if age is not None and query.as_of_ns - rows[-1].source_ns > age:
                raise FeatureUnavailable("STALE", f"{requirement.alias}源数据过旧")
            values[requirement.alias] = rows
            fingerprints[requirement.alias] = spec.fingerprint
        return FeatureSnapshot(query, generation, values, fingerprints)
