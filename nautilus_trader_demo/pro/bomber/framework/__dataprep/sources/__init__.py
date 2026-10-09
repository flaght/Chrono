"""可替换的参考资料来源；导入不加载数据库SDK、不联网。"""
from .base import (ReferenceBatch, ReferenceDataSource, ReferenceDataset,
                   ReferenceQuery, ReferenceSourceError)
from .dolphindb import DolphinDbReferenceConfig, DolphinDbReferenceSource
from .factory import MongoDbReferenceSource, MySqlReferenceSource, ReferenceSourceFactory
from .file import FileReferenceSource
from .policy import DataSourcePurpose, validate_source

__all__ = ["ReferenceBatch", "ReferenceDataSource", "ReferenceDataset", "ReferenceQuery",
           "ReferenceSourceError", "DolphinDbReferenceConfig", "DolphinDbReferenceSource",
           "MySqlReferenceSource", "MongoDbReferenceSource", "FileReferenceSource",
           "ReferenceSourceFactory", "DataSourcePurpose", "validate_source"]
