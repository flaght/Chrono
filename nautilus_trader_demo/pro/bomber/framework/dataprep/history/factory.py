"""历史行情Provider工厂；扩展后端只需注册，不改策略或恢复协调器。"""
from .base import HistoryProvider
from .file import FileHistoryProvider
from .dolphindb import DolphinDbHistoryProvider
from ..sources.policy import DataSourcePurpose


class MongoDbHistoryProvider(HistoryProvider):
    def __init__(self, config, **kwargs):
        self.config = config
    def open(self):
        raise NotImplementedError("MongoDB历史Provider尚未接入；可通过工厂注册替代实现")
    def read(self, request):
        raise NotImplementedError("MongoDB历史Provider尚未接入")
    def close(self):
        pass


class HistoryProviderFactory:
    _providers = {"file": FileHistoryProvider, "dolphindb": DolphinDbHistoryProvider, "mongodb": MongoDbHistoryProvider}

    @classmethod
    def backends(cls):
        return tuple(cls._providers)

    @classmethod
    def create_for(cls, purpose, backend, config, **kwargs):
        purpose = DataSourcePurpose(purpose)
        source = cls._providers.get(str(backend).strip().lower())
        if source is None:
            raise ValueError("历史Provider未注册")
        allowed = {DataSourcePurpose.OFFLINE_MARKET_HISTORY: {"file"},
            DataSourcePurpose.LIVE_MARKET_HISTORY: {"file", "database"},
            DataSourcePurpose.LIVE_RECOVERY: {"database"}}
        if purpose not in allowed or source.storage_kind not in allowed[purpose]:
            raise ValueError("用途不允许该历史Provider，禁止静默回退")
        return cls.create(backend, config, **kwargs)

    @classmethod
    def create(cls, backend, config, **kwargs):
        name = str(backend).strip().lower()
        if name not in cls._providers:
            raise ValueError(f"未知历史行情Provider: {name}")
        return cls._providers[name](config, **kwargs)

    @classmethod
    def register(cls, backend, provider_type, *, replace=False):
        name = str(backend).strip().lower()
        if not name or (name in cls._providers and not replace):
            raise ValueError("历史Provider名称为空或已注册")
        if not isinstance(provider_type, type) or not issubclass(provider_type, HistoryProvider):
            raise TypeError("工厂只接受HistoryProvider子类")
        cls._providers[name] = provider_type
