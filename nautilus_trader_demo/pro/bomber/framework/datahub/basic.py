"""期货与期权共用的条款校验和时间查询；不读取数据文件。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Generic, Iterable, Mapping, TypeVar

from .temporal import AsOfQuery, DataHub, InMemoryAsOfProvider, ReferenceRecord

T = TypeVar("T")


def missing(value: object) -> bool:
    return value is None or str(value).strip().lower() in {"", "nan", "nat", "none", "<na>"}


def text(value: object, name: str) -> str:
    if missing(value):
        raise ValueError(f"合约条款 {name} 不能为空")
    return str(value).strip()


def positive_decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"合约条款 {name} 必须为有限正数") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"合约条款 {name} 必须为有限正数")
    return result


def contract_date(value: object, name: str) -> date:
    if missing(value):
        raise ValueError(f"合约条款 {name} 不能为空")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip().split("T")[0].split(" ")[0])
    except ValueError as exc:
        raise ValueError(f"合约条款 {name} 日期无效: {value}") from exc


def field(row: Mapping[str, object], *names: str) -> object:
    """非空无效值不跳过；仅缺失值才使用后续兼容字段。"""
    for name in names:
        if name in row and not missing(row[name]):
            return row[name]
    return None


class BasicProvider(Generic[T]):
    """两类条款使用同一版本、发布时间、来源年龄和日终门控。"""

    dataset: str
    value_type: type[T]

    def __init__(self, records: Iterable[ReferenceRecord[T]]) -> None:
        rows = tuple(records)
        for row in rows:
            if row.dataset != self.dataset or not isinstance(row.value, self.value_type):
                raise ValueError("条款记录的数据集或值类型不匹配")
            if row.key != row.value.symbol:
                raise ValueError("条款记录的键与合约代码不一致")
        self._hub = DataHub({self.dataset: InMemoryAsOfProvider(rows)})

    def read(self, dataset: str, key: str, query: AsOfQuery) -> tuple[ReferenceRecord[T], ...]:
        return self._hub.get(dataset, key, query)

    def basic_at(self, symbol: str, query: AsOfQuery) -> T:
        return self.read(self.dataset, symbol, query)[-1].value
