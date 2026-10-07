"""不可变期权条款与按事件时间查询的参考资料接口。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Callable, Mapping

from .basic import (BasicProvider, missing as _missing, text as _text,
                    positive_decimal as _positive_decimal, contract_date as _date, field as _field)
from .temporal import ReferenceRecord


OPTION_BASIC_DATASET = "option_basic"


@dataclass(frozen=True)
class OptionBasic:
    """保留原始合约身份，不推断交易标的映射、价位或发布时间。"""

    symbol: str
    exchange: str
    currency: str
    underlying: str
    option_kind: str
    strike: Decimal
    multiplier: Decimal
    list_date: date
    last_trade_date: date
    expiry_date: date
    price_increment: Decimal | None = None
    exercise_style: str | None = None

    def __post_init__(self) -> None:
        for name in ("symbol", "exchange", "currency", "underlying"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        kind = _text(self.option_kind, "option_kind").upper()
        aliases = {"CO": "CALL", "C": "CALL", "CALL": "CALL", "认购": "CALL",
                   "PO": "PUT", "P": "PUT", "PUT": "PUT", "认沽": "PUT"}
        if kind not in aliases:
            raise ValueError(f"无法识别期权类型: {self.option_kind}")
        object.__setattr__(self, "option_kind", aliases[kind])
        for name in ("strike", "multiplier"):
            object.__setattr__(self, name, _positive_decimal(getattr(self, name), name))
        for name in ("list_date", "last_trade_date", "expiry_date"):
            object.__setattr__(self, name, _date(getattr(self, name), name))
        if not self.list_date <= self.last_trade_date <= self.expiry_date:
            raise ValueError("期权日期须满足上市日 <= 最后交易日 <= 到期日")
        if not _missing(self.price_increment):
            object.__setattr__(self, "price_increment",
                               _positive_decimal(self.price_increment, "price_increment"))
        else:
            object.__setattr__(self, "price_increment", None)
        style = None if _missing(self.exercise_style) else str(self.exercise_style).strip()
        object.__setattr__(self, "exercise_style", style)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> OptionBasic:
        """适配标准字段与源基础表字段，缺少价位时保持 None。"""
        return cls(
            symbol=_field(row, "symbol", "Code", "code"),
            exchange=_field(row, "exchange", "exchangeCD"),
            currency=_field(row, "currency", "currencyCD"),
            underlying=_field(row, "underlying", "varTicker"),
            option_kind=_field(row, "option_kind", "contractType"),
            strike=_field(row, "strike", "strikePrice"),
            multiplier=_field(row, "multiplier", "contMultNum"),
            list_date=_field(row, "list_date", "listDate"),
            last_trade_date=_field(row, "last_trade_date", "lastTradeDate"),
            expiry_date=_field(row, "expiry_date", "expDate"),
            price_increment=_field(row, "tickNum", "minChgPriceNum", "price_increment"),
            exercise_style=_field(row, "exercise_style", "exerciseStyle"),
        )


class OptionBasicProvider(BasicProvider[OptionBasic]):
    """期权条款版本查询，沿用与期货相同的时间校验。"""

    dataset = OPTION_BASIC_DATASET
    value_type = OptionBasic

    @classmethod
    def from_feather(
        cls, path: str | Path, *,
        record_factory: Callable[[OptionBasic, Mapping[str, object]], ReferenceRecord[OptionBasic]],
    ) -> OptionBasicProvider:
        """兼容入口；文件读取委托 dataprep，不自动解释 date 为发布时间。"""
        from bomber.framework.dataprep.basic import load_option_basic_provider

        return load_option_basic_provider(path, record_factory=record_factory)
