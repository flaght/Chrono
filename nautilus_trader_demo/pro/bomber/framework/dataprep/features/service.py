"""控制线程拉取特征，统一缓存供策略查询；错误不伪装成缺数据。"""
from dataclasses import replace
from threading import RLock

from bomber.framework.datahub.features import FeatureError
from bomber.framework.datahub.features.contracts import integer
from .base import FeatureSourceError


class FeatureRefreshService:
    def __init__(self, hub, providers, *, max_observation_age_ns=30_000_000_000):
        if hub.features is None or not providers:
            raise FeatureError("刷新服务须装配DataHub.features和Provider")
        integer(max_observation_age_ns, "max_observation_age_ns", 1)
        self.hub, self.providers = hub, dict(providers)  # (dataset_id,schema_version)->Provider
        if any(provider.registry is not hub.features.registry for provider in self.providers.values()):
            raise FeatureError("刷新服务所有组件须共享注册表")
        self.max_age_ns = max_observation_age_ns
        self._cursors = {}
        self._source_owners = {}
        self._clock_ns = -1
        self._lock = RLock()

    def refresh(self, requests, *, clock_ns):
        """调用方提供实际时钟；查询完成后才记录接收时间，避免慢查询提前可见。"""
        requests = tuple(requests)
        if not requests:
            raise FeatureError("刷新请求不能为空")
        with self._lock:
            started = clock_ns()
            integer(started, "刷新时钟")
            if started < self._clock_ns:
                raise FeatureSourceError("刷新时钟回退")
            records, cursors, scopes, owners = [], dict(self._cursors), [], dict(self._source_owners)
            try:
                for request in requests:
                    if request.as_of_ns > started:
                        raise FeatureError("数据库请求可见截止点不得晚于实际刷新开始")
                    identity = (request.dataset_id, request.schema_version)
                    provider = self.providers.get(identity)
                    if provider is None:
                        raise FeatureError("特征Provider未装配")
                    token = (identity, request.scope)
                    batch = provider.read_updates(request, cursors.get(token))
                    spec = self.hub.features.registry.get(*identity)
                    if batch.schema_fingerprint != spec.fingerprint:
                        raise FeatureError("Provider返回不兼容Schema")
                    observed = clock_ns()
                    integer(observed, "刷新完成时钟")
                    if observed < started or observed < self._clock_ns:
                        raise FeatureSourceError("查询过程中时钟回退")
                    if observed - started > self.max_age_ns:
                        raise FeatureSourceError("整批特征查询超过刷新时间预算")
                    for key in request.keys:
                        scope = (*identity, key)
                        if scope in owners and owners[scope] != batch.source_identity:
                            raise FeatureError("同一业务键来源身份变化，须显式重建会话")
                        owners[scope] = batch.source_identity
                        scopes.append(scope)
                    records.extend(replace(row, received_ns=max(observed, row.received_ns or 0))
                        if provider.storage_kind == "database" else row for row in batch.records)
                    cursors[token] = batch.cursor
                finished = clock_ns()
                integer(finished, "刷新完成时钟")
                if finished < observed or finished - started > self.max_age_ns:
                    raise FeatureSourceError("整批刷新超时或时钟回退")
                generation = self.hub.features.store.publish(records, health_updates=(
                    (scope, finished, finished + self.max_age_ns, None) for scope in set(scopes)))
            except Exception as error:
                try:
                    failed_at = clock_ns()
                    integer(failed_at, "失败观测时钟")
                except Exception:
                    failed_at = started
                failed_at = max(started, self._clock_ns, failed_at)
                code = "SCHEMA" if isinstance(error, FeatureError) else "SOURCE_ERROR"
                self.hub.features.store.publish((), health_updates=(
                    ((request.dataset_id, request.schema_version, key), failed_at, failed_at, code)
                    for request in requests for key in request.keys))
                self._clock_ns = failed_at
                raise
            self._cursors, self._source_owners = cursors, owners
            self._clock_ns = finished
            return generation
