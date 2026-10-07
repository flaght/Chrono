"""格式无关的DataHub时间契约与最小as-of查询实现。

这里不读取文件、不推导交易日、不返回持仓或活动订单。交易日由调用方的
行情日历提供；源适配器只需产生ReferenceRecord，未来可来自任意存储。
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Generic, Iterable, Mapping, Protocol, TypeVar


T = TypeVar("T")
ReferenceKey = tuple[str, str]


class DataHubUnavailable(LookupError):
    """当前决策时刻缺少足够且已发布的参考数据。"""


class FutureDataError(ValueError):
    """查询或Provider试图返回尚未发生/发布的数据。"""


class PublicationPolicy(str, Enum):
    INTRADAY = "intraday"
    DAY_END = "day_end"


@dataclass(frozen=True)
class ReferenceRecord(Generic[T]):
    """标准化的单条参考数据，value应为不可变值对象。"""

    dataset: str
    key: str
    value: T
    event_ns: int  # 数据生效/描述的事件时刻。
    available_ns: int  # 数据最早实际可被策略得知的时刻。
    source_ns: int  # 原始数据时刻；向前延续时用于计算真实年龄。
    source_trading_day: date | None = None
    policy: PublicationPolicy = PublicationPolicy.INTRADAY

    def __post_init__(self) -> None:
        if not self.dataset or not self.key:
            raise ValueError("数据集名称和键不能为空")
        if any(type(value) is not int for value in (self.event_ns, self.available_ns, self.source_ns)):
            raise TypeError("时间戳必须为整数纳秒")
        if min(self.event_ns, self.available_ns, self.source_ns) < 0:
            raise ValueError("时间戳不能为负")
        if self.source_ns > self.event_ns:
            raise ValueError("原始数据时刻不能晚于事件时刻")
        if not isinstance(self.policy, PublicationPolicy):
            raise TypeError("发布策略必须是PublicationPolicy")
        if self.policy is PublicationPolicy.DAY_END and self.source_trading_day is None:
            raise ValueError("日终数据必须携带来源交易日")


@dataclass(frozen=True)
class AsOfQuery:
    as_of_ns: int
    trading_day: date | None = None
    data_time_ns: int | None = None
    window: int = 1
    max_source_age_ns: int | None = None

    def __post_init__(self) -> None:
        if type(self.as_of_ns) is not int or (
            self.data_time_ns is not None and type(self.data_time_ns) is not int
        ):
            raise TypeError("查询时间必须为整数纳秒")
        if self.as_of_ns < 0 or self.data_time_ns is not None and (
            self.data_time_ns < 0 or self.data_time_ns > self.as_of_ns
        ):
            raise FutureDataError("数据查询时间不能晚于当前决策时间")
        if type(self.window) is not int or self.window < 1:
            raise ValueError("窗口长度必须大于零")
        if self.max_source_age_ns is not None and (
            type(self.max_source_age_ns) is not int or self.max_source_age_ns < 0
        ):
            raise ValueError("最大源数据年龄不能为负")

    @property
    def cutoff_ns(self) -> int:
        return self.as_of_ns if self.data_time_ns is None else self.data_time_ns


class AsOfProviderPort(Protocol):
    """文件、数据库或远程服务只需实现此读取协议。"""

    def read(self, dataset: str, key: str, query: AsOfQuery) -> tuple[ReferenceRecord, ...]: ...


class InMemoryAsOfProvider:
    """首阶段夹具Provider；先按event定位，再应用交易日发布规则。"""

    def __init__(self, records: Iterable[ReferenceRecord]) -> None:
        grouped: dict[ReferenceKey, list[ReferenceRecord]] = {}
        for record in records:
            grouped.setdefault((record.dataset, record.key), []).append(record)
        self._rows: dict[ReferenceKey, tuple[ReferenceRecord, ...]] = {}
        self._times: dict[ReferenceKey, tuple[int, ...]] = {}
        for key, values in grouped.items():
            ordered = tuple(sorted(values, key=lambda item: item.event_ns))
            if any(a.event_ns == b.event_ns for a, b in zip(ordered, ordered[1:])):
                raise ValueError(f"同一数据键存在重复event_ns: {key}")
            self._rows[key] = ordered
            self._times[key] = tuple(item.event_ns for item in ordered)

    def read(self, dataset: str, key: str, query: AsOfQuery) -> tuple[ReferenceRecord, ...]:
        identity = (dataset, key)
        end = bisect_right(self._times.get(identity, ()), query.cutoff_ns)
        eligible = []
        for item in self._rows.get(identity, ())[:end]:
            if item.policy is PublicationPolicy.DAY_END:
                if query.trading_day is None:
                    raise ValueError("查询日终数据必须提供行情对应的交易日")
                if item.source_trading_day >= query.trading_day:
                    continue  # 当日日终表不可用于当日盘中，不能减一个自然日。
            eligible.append(item)
        if len(eligible) < query.window:
            raise DataHubUnavailable(f"{identity}在当前时点不足{query.window}条历史数据")
        # 最新生效版本未发布时fail-closed，不悄悄退回旧角色/旧因子。
        return tuple(eligible[-query.window:])


@dataclass(frozen=True)
class DataHubSnapshot:
    query: AsOfQuery
    values: Mapping[ReferenceKey, tuple[ReferenceRecord, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def latest(self, dataset: str, key: str) -> ReferenceRecord:
        return self.values[(dataset, key)][-1]


class DataHub:
    """格式无关的最小数据查询门面；按数据集装配Provider。"""

    def __init__(self, providers: Mapping[str, AsOfProviderPort]) -> None:
        if not providers or any(not name for name in providers):
            raise ValueError("DataHub至少需要一个命名Provider")
        self._providers = MappingProxyType(dict(providers))

    def get(self, dataset: str, key: str, query: AsOfQuery) -> tuple[ReferenceRecord, ...]:
        try:
            provider = self._providers[dataset]
        except KeyError as exc:
            raise DataHubUnavailable(f"未注册参考数据集: {dataset}") from exc
        rows = provider.read(dataset, key, query)
        if len(rows) != query.window:
            raise DataHubUnavailable(f"{dataset}/{key}窗口数据不足")
        previous_event = -1
        for item in rows:
            if (item.dataset, item.key) != (dataset, key):
                raise ValueError("Provider返回了其他数据集或键")
            if item.event_ns <= previous_event:
                raise ValueError("Provider返回顺序必须按event_ns严格递增")
            previous_event = item.event_ns
            if item.event_ns > query.cutoff_ns or item.available_ns > query.as_of_ns or item.source_ns > query.as_of_ns:
                raise FutureDataError(f"{dataset}/{key}包含决策时不可见的数据")
            if item.policy is PublicationPolicy.DAY_END:
                if query.trading_day is None:
                    raise ValueError("查询日终数据必须提供行情对应的交易日")
                if item.source_trading_day >= query.trading_day:
                    raise FutureDataError(f"{dataset}/{key}当日日终数据不能当日使用")
            if (query.max_source_age_ns is not None
                    and query.as_of_ns - item.source_ns > query.max_source_age_ns):
                raise DataHubUnavailable(f"{dataset}/{key}原始数据过旧")
        return tuple(rows)

    def snapshot(self, requests: Iterable[ReferenceKey], query: AsOfQuery) -> DataHubSnapshot:
        keys = tuple(requests)
        if len(set(keys)) != len(keys):
            raise ValueError("同一快照不能重复请求数据键")
        values = {identity: self.get(*identity, query) for identity in keys}
        return DataHubSnapshot(query, values)
