"""来源选择在装配层；Live只允许数据库，离线只允许文件。"""
from .base import FeatureProvider
from .database import DolphinDbFeatureProvider, MongoDbFeatureProvider
from .file import FileFeatureProvider
from .raw import DolphinDbRawFeatureProvider
from ..sources.policy import DataSourcePurpose


class FeatureProviderFactory:
    _providers = {"file": FileFeatureProvider, "dolphindb": DolphinDbFeatureProvider,
        "mongodb": MongoDbFeatureProvider, "dolphindb-raw": DolphinDbRawFeatureProvider}

    @classmethod
    def register(cls, name, provider_type, *, replace=False):
        name = str(name).strip().lower()
        if not name or name in cls._providers and not replace:
            raise ValueError("特征来源名称为空或重复")
        if not isinstance(provider_type, type) or not issubclass(provider_type, FeatureProvider):
            raise TypeError("特征来源须实现FeatureProvider")
        cls._providers[name] = provider_type

    @classmethod
    def create_for(cls, purpose, backend, config, registry, **kwargs):
        purpose = DataSourcePurpose(purpose)
        provider = cls._providers.get(str(backend).strip().lower())
        if provider is None:
            raise ValueError("特征Provider未注册")
        allowed = {DataSourcePurpose.OFFLINE_FEATURE: "file", DataSourcePurpose.LIVE_FEATURE: "database",
            DataSourcePurpose.FEATURE_EXPORT: "database"}
        if purpose not in allowed or provider.storage_kind != allowed[purpose]:
            raise ValueError("特征来源不符合用途；禁止隐式切换来源")
        return provider(config, registry, **kwargs)
