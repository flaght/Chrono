"""Nautilus通用模拟交易场所配置。

Profile只描述交易场所规则，不拥有回测时钟、行情和订单生命周期。
``NautilusSimExecutionBackend``会把这里生成的参数传给
``BacktestEngine.add_venue``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from bomber.core.rust.model import OtoTriggerMode
from bomber.model import Currency, Money, Venue
from bomber.model.enums import AccountType, BookType, OmsType
from bomber.model.identifiers import InstrumentId


@dataclass(frozen=True)
class GenericVenueProfile:
    """可直接映射到Nautilus ``BacktestEngine.add_venue`` 的通用Profile。

    这里仅包含Nautilus已经公开支持的通用规则。CTP平今/平昨、交易日历，
    Binance资金费率等扩展规则由后续专用Profile组合实现。
    """

    profile_id: str
    venue: Venue | str
    oms_type: OmsType
    account_type: AccountType
    starting_balances: Sequence[Money]
    base_currency: Currency | None = None
    default_leverage: Decimal | None = None
    leverages: Mapping[InstrumentId, Decimal] = field(default_factory=dict)
    margin_model: Any = None
    modules: Sequence[Any] = field(default_factory=tuple)
    fill_model: Any = None
    fee_model: Any = None
    latency_model: Any = None
    book_type: BookType = BookType.L1_MBP
    routing: bool = False
    reject_stop_orders: bool = True
    support_gtd_orders: bool = True
    support_contingent_orders: bool = True
    oto_trigger_mode: OtoTriggerMode = OtoTriggerMode.PARTIAL
    use_position_ids: bool = True
    use_random_ids: bool = False
    use_reduce_only: bool = True
    use_message_queue: bool = True
    use_market_order_acks: bool = False
    bar_execution: bool = True
    bar_adaptive_high_low_ordering: bool = False
    trade_execution: bool = True
    liquidity_consumption: bool = False
    queue_position: bool = False
    allow_cash_borrowing: bool = False
    frozen_account: bool = False
    price_protection_points: Any = None
    settlement_prices: Mapping[InstrumentId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id不能为空")
        venue = self.venue if isinstance(self.venue, Venue) else Venue(self.venue)
        balances = tuple(self.starting_balances)
        if not balances:
            raise ValueError("starting_balances不能为空")
        if self.default_leverage is not None and self.default_leverage <= 0:
            raise ValueError("default_leverage必须大于零")
        if any(value <= 0 for value in self.leverages.values()):
            raise ValueError("instrument leverage必须大于零")
        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "starting_balances", balances)
        object.__setattr__(self, "leverages", MappingProxyType(dict(self.leverages)))
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(
            self,
            "settlement_prices",
            MappingProxyType(dict(self.settlement_prices)),
        )

    def build_backend_config(self) -> Mapping[str, Any]:
        """返回一份可直接展开给 ``BacktestEngine.add_venue`` 的新字典。"""

        return {
            "venue": self.venue,
            "oms_type": self.oms_type,
            "account_type": self.account_type,
            "starting_balances": list(self.starting_balances),
            "base_currency": self.base_currency,
            "default_leverage": self.default_leverage,
            "leverages": dict(self.leverages) or None,
            "margin_model": self.margin_model,
            "modules": list(self.modules) or None,
            "fill_model": self.fill_model,
            "fee_model": self.fee_model,
            "latency_model": self.latency_model,
            "book_type": self.book_type,
            "routing": self.routing,
            "reject_stop_orders": self.reject_stop_orders,
            "support_gtd_orders": self.support_gtd_orders,
            "support_contingent_orders": self.support_contingent_orders,
            "oto_trigger_mode": self.oto_trigger_mode,
            "use_position_ids": self.use_position_ids,
            "use_random_ids": self.use_random_ids,
            "use_reduce_only": self.use_reduce_only,
            "use_message_queue": self.use_message_queue,
            "use_market_order_acks": self.use_market_order_acks,
            "bar_execution": self.bar_execution,
            "bar_adaptive_high_low_ordering": self.bar_adaptive_high_low_ordering,
            "trade_execution": self.trade_execution,
            "liquidity_consumption": self.liquidity_consumption,
            "queue_position": self.queue_position,
            "allow_cash_borrowing": self.allow_cash_borrowing,
            "frozen_account": self.frozen_account,
            "price_protection_points": self.price_protection_points,
            "settlement_prices": dict(self.settlement_prices) or None,
        }
