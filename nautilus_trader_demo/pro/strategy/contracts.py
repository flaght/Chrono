"""与具体数据源无关的策略及执行数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from market.basic.base import DataType, InstrumentId


class RuntimeMode(str, Enum):
    """运行时使用的时间环境。具体能力由RuntimePort实现决定。"""

    HISTORICAL = "historical"
    LIVE = "live"


class TargetUpdateMode(str, Enum):
    """目标更新方式。

    ``REPLACE`` 表示本次提交是策略目标作用域内的完整快照，未出现的旧目标
    将被移除；``PATCH`` 只覆盖本次出现的目标键。
    """

    REPLACE = "REPLACE"
    PATCH = "PATCH"


@dataclass(frozen=True)
class DataBinding:
    """把策略内部的数据键绑定到一个标准行情订阅。"""

    data_key: str
    feed_id: str
    instrument_id: InstrumentId
    data_type: DataType
    bar_spec: str | None = None

    def __post_init__(self) -> None:
        if not self.data_key.strip():
            raise ValueError("data_key 不能为空")
        if not self.feed_id.strip():
            raise ValueError("feed_id 不能为空")
        if self.data_type in (DataType.BAR, DataType.CUSTOM_BAR):
            if self.bar_spec is None:
                raise ValueError("Bar 数据绑定必须提供 bar_spec")
            object.__setattr__(self, "bar_spec", self.bar_spec.upper())


@dataclass(frozen=True)
class ExecutionRoute:
    """把策略内部的目标键解析到交易客户端和实际交易标的。"""

    target_key: str
    client_id: str
    instrument_id: InstrumentId

    def __post_init__(self) -> None:
        if not self.target_key.strip():
            raise ValueError("target_key 不能为空")
        if not self.client_id.strip():
            raise ValueError("client_id 不能为空")


@dataclass(frozen=True)
class TargetPortfolio:
    """策略使用逻辑目标键表达的目标组合，不包含交易客户端 API。"""

    strategy_id: str
    revision: int
    ts_event: int
    targets: Mapping[str, Decimal]
    execution_policy: str = "DIRECT"
    deadline_ns: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    update_mode: TargetUpdateMode = TargetUpdateMode.REPLACE

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id 不能为空")
        if self.revision < 1:
            raise ValueError("revision 必须为正整数")
        if self.ts_event < 0:
            raise ValueError("ts_event 不能为负数")
        normalized = {
            key: _decimal(quantity)
            for key, quantity in self.targets.items()
        }
        if not normalized:
            raise ValueError("targets 不能为空")
        if any(not key.strip() for key in normalized):
            raise ValueError("target_key 不能为空")
        object.__setattr__(self, "update_mode", TargetUpdateMode(self.update_mode))
        object.__setattr__(self, "targets", MappingProxyType(normalized))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ExecutionRequest:
    """已经解析为实际交易标的、发送给单个交易客户端的请求。"""

    strategy_id: str
    revision: int
    client_id: str
    ts_event: int
    targets: Mapping[InstrumentId, Decimal]
    execution_policy: str
    deadline_ns: int | None = None
    logical_targets: Mapping[str, Decimal] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(
            self,
            "logical_targets",
            MappingProxyType(dict(self.logical_targets)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _decimal(value: Decimal | int | float | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))
