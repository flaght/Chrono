"""为单角色在线参考提供控制线程缓存，保留原适配器的读取接口。"""

from datetime import datetime
import time

from bomber.framework.datahub.sector_roles import SectorDataUnavailable
from .live_cache import LiveReferenceCache
from .live_role import FileRoleReferences


class CachedRoleReferences:
    refresh_on_event = False

    def __init__(self, references):
        self.source = references
        for key in ("product", "role", "trading_day", "started_ns", "paths",
                    "factor_date_basis", "factor_availability"):
            setattr(self, key, getattr(references, key))
        self._published = LiveReferenceCache()
        self.refresh()

    @property
    def publication_lock(self):
        return self._published.publication_lock

    @publication_lock.setter
    def publication_lock(self, value):
        self._published.publication_lock = value

    def refresh(self):
        def load():
            assignment = self.source.snapshot(time.time_ns())
            return assignment, self.source.spec, self.source.manifest
        self._published.refresh(load)

    @property
    def spec(self):
        return self._published.value[1]

    @property
    def version(self):
        assignment = self.snapshot(time.time_ns())
        return (self.spec, assignment.instrument(self.product, self.role),
                assignment.factor(self.product, self.role))

    @property
    def manifest(self):
        return self._published.manifest

    @property
    def factor_date(self):
        return datetime.fromisoformat(self.manifest["factor_date"]).date()

    @property
    def instrument_id(self):
        return FileRoleReferences.instrument_id.fget(self)

    def instrument_meta(self):
        return FileRoleReferences.instrument_meta(self)

    def snapshot(self, as_of_ns):
        try:
            return self._published.snapshot(as_of_ns)
        except RuntimeError as error:
            # 未生效的参考仍遵循单角色策略已有的缺失资料处理接口。
            if str(error) == "当前参考版本在Bar时刻尚不可见":
                raise SectorDataUnavailable(str(error)) from error
            raise
