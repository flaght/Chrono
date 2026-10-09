"""参考资料数据源工厂；未实现后端不回退到其他数据库或文件。"""
from .base import ReferenceDataSource
from .dolphindb import DolphinDbReferenceSource
from .file import FileReferenceSource
from .policy import validate_source


class MySqlReferenceSource(ReferenceDataSource):
    def __init__(self, config):
        self._config = config

    def open(self):
        raise NotImplementedError("MySQL 参考资料适配器尚未实现")

    def read(self, dataset, query):
        raise NotImplementedError("MySQL 参考资料适配器尚未实现")

    def close(self):
        pass


class MongoDbReferenceSource(ReferenceDataSource):
    def __init__(self, config):
        self._config = config

    def open(self):
        raise NotImplementedError("MongoDB 参考资料适配器尚未实现")

    def read(self, dataset, query):
        raise NotImplementedError("MongoDB 参考资料适配器尚未实现")

    def close(self):
        pass


class ReferenceSourceFactory:
    _sources = {"file": FileReferenceSource, "dolphindb": DolphinDbReferenceSource,
                "mysql": MySqlReferenceSource, "mongodb": MongoDbReferenceSource}

    @classmethod
    def create_for(cls, purpose, backend, config, **kwargs):
        """先拒绝不允许的来源，再构造；构造过程不打开SDK会话。"""
        validate_source(backend, purpose)
        return cls.create(backend, config, **kwargs)

    @classmethod
    def create(cls, backend, config, **kwargs):
        name = str(backend).strip().lower()
        if name not in cls._sources:
            raise ValueError(f"未知参考资料数据源: {name}")
        return cls._sources[name](config, **kwargs)

    @classmethod
    def register(cls, backend, source_type):
        name = str(backend).strip().lower()
        if not name or name in cls._sources:
            raise ValueError("数据源名称为空或已注册")
        if not isinstance(source_type, type) or not issubclass(source_type, ReferenceDataSource):
            raise TypeError("工厂只接受 ReferenceDataSource 子类")
        cls._sources[name] = source_type
