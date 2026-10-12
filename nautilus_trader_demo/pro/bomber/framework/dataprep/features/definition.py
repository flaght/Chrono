"""JSON特征清单装配；新增已计算特征只增声明，不增Provider代码。"""
from dataclasses import dataclass, field
from collections.abc import Mapping
import json
from pathlib import Path
from types import MappingProxyType

from bomber.framework.datahub.features import DatasetRegistry, FeatureError
from bomber.framework.datahub.features.schema import dataset_from_definition, only_keys
from .raw import DolphinDbRawConfig

DEFAULT_FEATURE_CONFIG = Path(__file__).with_name("configs") / "features.json"


@dataclass(frozen=True)
class FeatureBinding:
    spec: object
    backend: str
    source_config: object
    bar_projection: object = None
    provider_options: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "provider_options", MappingProxyType(dict(self.provider_options)))
        if self.bar_projection is not None:
            only_keys(self.bar_projection, ("bar_spec", "fields"), "Bar投影")
            mapping = dict(self.bar_projection["fields"])
            if set(mapping) != {"open", "high", "low", "close", "volume"}:
                raise FeatureError("Bar投影须显式映射真实OHLCV五个字段")
            kinds = {field.name: field.kind for field in self.spec.payload_fields}
            if any(name not in kinds or kinds[name] not in {"decimal", "float", "int"} for name in mapping.values()):
                raise FeatureError("Bar投影须绑定Schema中的数值字段")
            object.__setattr__(self, "bar_projection", MappingProxyType({
                "bar_spec": self.bar_projection["bar_spec"], "fields": MappingProxyType(mapping)}))


class FeatureBundle:
    def __init__(self, registry, bindings):
        self.registry = registry
        self.bindings = MappingProxyType({(b.spec.dataset_id, b.spec.schema_version): b for b in bindings})

    def binding(self, dataset_id, schema_version=1):
        try:
            return self.bindings[(dataset_id, schema_version)]
        except KeyError:
            raise FeatureError("清单未声明该数据集/版本") from None

    def provider(self, dataset_id, schema_version=1, *, purpose=None, **kwargs):
        from .factory import FeatureProviderFactory
        from ..sources.policy import DataSourcePurpose
        binding = self.binding(dataset_id, schema_version)
        default_purpose = DataSourcePurpose.OFFLINE_FEATURE if binding.backend == "file" else DataSourcePurpose.LIVE_FEATURE
        return FeatureProviderFactory.create_for(purpose or default_purpose,
            binding.backend, binding.source_config, self.registry, **{**binding.provider_options, **kwargs})


def source_from_definition(source, connection, spec):
    """只装配白名单存储协议；连接凭据由调用方注入，不放数据集JSON中。"""
    from .codec import FeatureColumnMapping
    from .database import DatasetTable, DolphinDbFeatureConfig, MongoDbFeatureConfig
    if not isinstance(source, dict):
        raise FeatureError("来源定义须为声明对象")
    backend = source.get("backend")
    if backend == "dolphindb-raw":
        config = DolphinDbRawConfig.from_definition(source,
            connection.get("dolphindb") if isinstance(connection, Mapping) else connection, spec)
        return backend, config, {}
    if backend not in {"file", "dolphindb", "mongodb"}:
        raise FeatureError("特征来源须选择已支持的存储协议/格式")
    only_keys(source, ("backend", "path", "database", "table", "mapping", "require_manifest", "source_id"), "来源")
    mapping_def = source.get("mapping", {})
    only_keys(mapping_def, ("columns", "timestamp_label"), "标准列映射")
    mapping = FeatureColumnMapping(**mapping_def)
    if backend == "file":
        if set(source) - {"backend", "path", "mapping", "require_manifest"}:
            raise FeatureError("文件来源不能混用数据库配置")
        required = source.get("require_manifest", True)
        if type(required) is not bool or type(source["path"]) is not str or not source["path"]:
            raise FeatureError("文件路径及清单开关无效")
        return backend, Path(source["path"]), {"mapping": mapping, "require_manifest": required}
    if set(source) - {"backend", "database", "table", "mapping", "source_id"}:
        raise FeatureError("数据库来源不能混用文件配置")
    tables = (DatasetTable(spec.dataset_id, spec.schema_version, source["table"], mapping),)
    if backend == "dolphindb":
        from ..sources.dolphindb import DolphinDbReferenceConfig
        c = connection.get("dolphindb") if isinstance(connection, Mapping) else connection
        if not isinstance(c, DolphinDbReferenceConfig):
            raise FeatureError("DolphinDB连接须通过调用方注入ReferenceConfig")
        config = DolphinDbFeatureConfig(c.host, c.port, c.username, c.password, source["database"], tables,
            timeout_seconds=c.read_timeout_seconds, enable_ssl=c.enable_ssl)
    else:
        c = connection.get("mongodb") if isinstance(connection, Mapping) else None
        if not isinstance(c, Mapping) or not c.get("uri"):
            raise FeatureError("MongoDB连接须通过connections.mongodb注入URI")
        config = MongoDbFeatureConfig(c["uri"], source["database"], tables, source["source_id"],
            timeout_seconds=c.get("timeout_seconds", 15))
    return backend, config, {}


def bundle_from_definition(definition, connection, *, calendars=None, base_dir=None):
    only_keys(definition, ("format_version", "datasets"), "特征清单")
    if type(definition.get("format_version")) is not int or definition["format_version"] != 1 \
            or not isinstance(definition.get("datasets"), list) or not definition["datasets"]:
        raise FeatureError("特征清单须有format_version=1及非空datasets")
    registry, bindings = DatasetRegistry(), []
    try:
        for dataset in definition["datasets"]:
            only_keys(dataset, ("schema", "source", "bar_projection"), "数据集定义")
            spec = dataset_from_definition(dataset["schema"], calendars=calendars)
            registry.register(spec)
            source = dataset["source"]
            backend, config, options = source_from_definition(source, connection, spec)
            if backend == "file" and not config.is_absolute() and base_dir is not None:
                config = Path(base_dir) / config
            bindings.append(FeatureBinding(spec, backend, config, dataset.get("bar_projection"), options))
    except (KeyError, TypeError) as error:
        raise FeatureError("特征清单缺字段或结构无效") from None
    return FeatureBundle(registry, bindings)


def load_feature_bundle(path, connection, *, calendars=None):
    try:
        definition = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise FeatureError("特征配置文件不可读取或JSON格式无效") from None
    return bundle_from_definition(definition, connection, calendars=calendars, base_dir=Path(path).resolve().parent)
