"""参考资料数据源契约。查询日期是源表日期，不隐含发布或交易日推导。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping


class ReferenceDataset(str, Enum):
    FUTURES_BASIC = "futures_basic"
    OPTIONS_BASIC = "options_basic"
    ADJUSTMENT_FACTORS = "adjustment_factors"
    CONTRACT_STRUCTURE = "contract_structure"


class ReferenceSourceError(RuntimeError):
    """数据源读取失败；不表示数据为空，不包含底层异常中的连接凭据。"""


@dataclass(frozen=True)
class ReferenceQuery:
    products: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    start_date: date | None = None
    end_date: date | None = None
    active_on: date | None = None

    def __post_init__(self):
        products = tuple(dict.fromkeys(str(p).strip().upper() for p in self.products))
        symbols = tuple(dict.fromkeys(str(s).strip().upper() for s in self.symbols))
        if any(re.fullmatch(r"[A-Z]+", p) is None for p in products):
            raise ValueError("品种代码须为字母")
        if any(re.fullmatch(r"[A-Z0-9]+(?:-[CP]-[0-9]+(?:\.[0-9]+)?)?", s) is None for s in symbols):
            raise ValueError("查询须使用真实合约代码，不带交易所后缀")
        if not products and not symbols:
            raise ValueError("查询须明确品种或合约范围")
        for value in (self.start_date, self.end_date, self.active_on):
            if value is not None and type(value) is not date:
                raise TypeError("查询日期须为 datetime.date")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("查询起始日期晚于结束日期")
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "symbols", symbols)


@dataclass(frozen=True)
class ReferenceBatch:
    dataset: ReferenceDataset
    rows: tuple[Mapping[str, object], ...]
    columns: tuple[str, ...]
    source: str
    fingerprint: str = field(init=False)

    def __post_init__(self):
        rows = tuple(MappingProxyType(dict(row)) for row in self.rows)
        object.__setattr__(self, "rows", rows)
        # 行序不作为版本变化；指纹只是读取内容证据，不是上游发布版本。
        encoded = sorted(json.dumps(dict(row), sort_keys=True, default=str,
                                    ensure_ascii=False, allow_nan=False) for row in rows)
        payload = json.dumps((self.dataset.value, sorted(self.columns), encoded), ensure_ascii=False)
        object.__setattr__(self, "fingerprint", hashlib.sha256(payload.encode()).hexdigest())

    def frame(self):
        import pandas as pd
        return pd.DataFrame([dict(row) for row in self.rows], columns=self.columns)


class ReferenceDataSource(ABC):
    """所有数据源子类返回同一标准字段；构造和导入不联网。"""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def read(self, dataset: ReferenceDataset, query: ReferenceQuery) -> ReferenceBatch: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def futures_basic(self, query):
        return self.read(ReferenceDataset.FUTURES_BASIC, query)

    def options_basic(self, query):
        return self.read(ReferenceDataset.OPTIONS_BASIC, query)

    def adjustment_factors(self, query):
        return self.read(ReferenceDataset.ADJUSTMENT_FACTORS, query)

    def contract_structure(self, query):
        return self.read(ReferenceDataset.CONTRACT_STRUCTURE, query)
