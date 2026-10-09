"""文件参考资料适配器；既有场景加载入口保持独立兼容。"""
from pathlib import Path
from types import MappingProxyType

from ..session import read_feather
from .base import ReferenceDataSource, ReferenceDataset, ReferenceQuery, ReferenceSourceError
from .normalize import normalize_frame


class FileReferenceSource(ReferenceDataSource):
    def __init__(self, paths):
        self.paths = MappingProxyType({ReferenceDataset(key): Path(value).expanduser().resolve()
                                       for key, value in paths.items()})
        self._opened = False

    def open(self):
        self._opened = True

    def close(self):
        self._opened = False

    def read(self, dataset, query):
        dataset = ReferenceDataset(dataset)
        if not isinstance(query, ReferenceQuery):
            raise TypeError("查询需要 ReferenceQuery")
        if not self._opened:
            raise ReferenceSourceError("须先显式 open() 数据源")
        if dataset not in self.paths:
            raise ValueError(f"未配置 {dataset.value} 文件")
        path = self.paths[dataset]
        return normalize_frame(read_feather(path), dataset, query, source=str(path))
