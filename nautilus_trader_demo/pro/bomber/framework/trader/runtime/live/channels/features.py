"""公共特征查询就绪检查；只读缓存，不按特征名称创建Channel。"""
import math
import time

from ..contracts import ChannelIdentity, ReadinessIssue, ReadinessResult


class FeatureQueryChannel:
    def __init__(self, cache, *, clock_ns=None, max_query_age_seconds=30, domain="market"):
        self.cache = cache
        self.clock_ns = clock_ns or time.time_ns
        if not math.isfinite(max_query_age_seconds) or max_query_age_seconds <= 0:
            raise ValueError("特征查询就绪时效须为有限正数")
        self.max_age_ns = int(max_query_age_seconds*1e9)
        self.identity = ChannelIdentity(domain, cache.manifest["backend"], cache.source_id)

    def check(self):
        data, now = self.cache.manifest, self.clock_ns()
        issues = []
        if not data["connected"] or data["failure"]:
            issues.append(ReadinessIssue(self.identity, "FEATURE_SOURCE_ERROR", data["failure"] or "特征缓存未连接"))
        elif data["last_query_ns"] is None or not 0 <= now-data["last_query_ns"] <= self.max_age_ns:
            issues.append(ReadinessIssue(self.identity, "FEATURE_QUERY_STALE", "特征查询未就绪或过期"))
        elif not data["cached_records"]:
            issues.append(ReadinessIssue(self.identity, "FEATURE_MISSING", "指定区间尚无可见特征"))
        return ReadinessResult((self.identity,), tuple(issues))
