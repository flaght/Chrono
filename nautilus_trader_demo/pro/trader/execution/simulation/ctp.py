"""CTP期货基础模拟规则。

本模块只提供可重复验证的基础回测近似。CTP真实柜台的双向持仓、今昨仓、
平今/平昨手续费、交易日切换和结算规则属于后续高保真Profile，不在这里隐式模拟。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from bomber.backtest.models import PerContractFeeModel, StandardMarginModel
from bomber.model import Money, Price, Quantity, Symbol, Venue
from bomber.model.currencies import CNY
from bomber.model.enums import AccountType, AssetClass, OmsType
from bomber.model.identifiers import InstrumentId
from bomber.model.instruments import FuturesContract

from trader.execution.simulation.profile import GenericVenueProfile


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True)
class CtpFuturesBasicProfile:
    """CTP期货的基础净持仓回测Profile。

    ``NETTING`` 是E7的有意限制，只用于验证基础撮合、合约乘数、保证金和固定
    每手手续费。需要同时持有多空仓，或区分平今/平昨时，必须使用后续E10的
    高保真Profile，不能继续使用本类。
    """

    profile_id: str = "ctp-futures-basic"
    starting_balance: Decimal | int | float | str = Decimal("1000000")
    commission_per_contract: Decimal | int | float | str = Decimal("1")
    venue: Venue = field(default_factory=lambda: Venue("SHFE"))

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id不能为空")
        balance = _decimal(self.starting_balance)
        commission = _decimal(self.commission_per_contract)
        if balance <= 0:
            raise ValueError("starting_balance必须大于零")
        if commission < 0:
            raise ValueError("commission_per_contract不能为负数")
        object.__setattr__(self, "starting_balance", balance)
        object.__setattr__(self, "commission_per_contract", commission)

    def build_backend_config(self) -> Mapping[str, Any]:
        """构造基础CTP模拟Venue配置。"""

        return GenericVenueProfile(
            profile_id=self.profile_id,
            venue=self.venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(self.starting_balance, CNY)],
            base_currency=CNY,
            default_leverage=Decimal(1),
            margin_model=StandardMarginModel(),
            fee_model=PerContractFeeModel(Money(self.commission_per_contract, CNY)),
            use_position_ids=False,
            use_reduce_only=True,
            use_market_order_acks=True,
            bar_execution=True,
            trade_execution=True,
        ).build_backend_config()

    def make_instrument(
        self,
        symbol: str,
        *,
        underlying: str,
        price_precision: int,
        price_increment: Decimal | int | float | str,
        multiplier: Decimal | int | float | str,
        activation_ns: int,
        expiration_ns: int,
        margin_init: Decimal | int | float | str,
        margin_maint: Decimal | int | float | str,
    ) -> FuturesContract:
        """按指定交易规则创建一个CTP期货合约。"""

        normalized = symbol.strip()
        normalized_underlying = underlying.strip()
        price_step = _decimal(price_increment)
        contract_multiplier = _decimal(multiplier)
        initial_rate = _decimal(margin_init)
        maintenance_rate = _decimal(margin_maint)
        if not normalized or not normalized_underlying:
            raise ValueError("symbol和underlying不能为空")
        if price_precision < 0:
            raise ValueError("price_precision不能为负数")
        if price_step <= 0 or contract_multiplier <= 0:
            raise ValueError("price_increment和multiplier必须大于零")
        if activation_ns < 0 or expiration_ns <= activation_ns:
            raise ValueError("expiration_ns必须晚于activation_ns")
        if initial_rate <= 0 or initial_rate > 1:
            raise ValueError("margin_init必须在(0, 1]范围内")
        if maintenance_rate <= 0 or maintenance_rate > initial_rate:
            raise ValueError("margin_maint必须大于零且不高于margin_init")

        instrument_id = InstrumentId.from_str(f"{normalized}.{self.venue}")
        return FuturesContract(
            instrument_id=instrument_id,
            raw_symbol=Symbol(normalized),
            asset_class=AssetClass.COMMODITY,
            currency=CNY,
            price_precision=price_precision,
            # 数据表常把整数tick写成1.0；Price.from_str("1.0")会保留1位
            # 精度，与price_precision=0冲突。只去掉无意义的尾随零。
            price_increment=Price.from_str(format(price_step.normalize(), "f")),
            multiplier=Quantity.from_str(str(contract_multiplier)),
            lot_size=Quantity.from_int(1),
            underlying=normalized_underlying,
            activation_ns=activation_ns,
            expiration_ns=expiration_ns,
            margin_init=initial_rate,
            margin_maint=maintenance_rate,
            maker_fee=Decimal(0),
            taker_fee=Decimal(0),
            exchange=str(self.venue),
            ts_event=0,
            ts_init=0,
            info={"market_type": "CTP_FUTURES_BASIC", "profile_id": self.profile_id},
        )
