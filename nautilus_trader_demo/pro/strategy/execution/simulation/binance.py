"""Binance USDT本位永续合约模拟规则。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from bomber.backtest.models import MakerTakerFeeModel, StandardMarginModel
from bomber.model import Money, Price, Quantity, Symbol, Venue
from bomber.model.currencies import USDT
from bomber.model.enums import AccountType, AssetClass, OmsType
from bomber.model.identifiers import InstrumentId
from bomber.model.instruments import PerpetualContract
from bomber.model.objects import Currency

from strategy.execution.simulation.profile import GenericVenueProfile


def _decimal(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True)
class BinanceUsdtFuturesProfile:
    """Binance USDT本位单向持仓模式的可配置模拟Profile。

    手续费率、杠杆和维持保证金率是回测参数，不在框架中宣称为交易所当前费率。
    调用方应按账户等级和历史时期显式覆盖这些值。
    """

    profile_id: str = "binance-usdt-futures"
    starting_balance: Decimal | int | float | str = Decimal("100000")
    leverage: Decimal | int | float | str = Decimal("10")
    maintenance_margin_rate: Decimal | int | float | str = Decimal("0.005")
    maker_fee: Decimal | int | float | str = Decimal("0.0002")
    taker_fee: Decimal | int | float | str = Decimal("0.0005")
    venue: Venue = field(default_factory=lambda: Venue("BINANCE"))

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id不能为空")
        balance = _decimal(self.starting_balance)
        leverage = _decimal(self.leverage)
        maintenance = _decimal(self.maintenance_margin_rate)
        maker_fee = _decimal(self.maker_fee)
        taker_fee = _decimal(self.taker_fee)
        if balance <= 0:
            raise ValueError("starting_balance必须大于零")
        if leverage <= 0:
            raise ValueError("leverage必须大于零")
        if maintenance <= 0 or maintenance > Decimal(1) / leverage:
            raise ValueError("maintenance_margin_rate必须大于零且不高于初始保证金率")
        if taker_fee < 0:
            raise ValueError("taker_fee不能为负数")
        object.__setattr__(self, "starting_balance", balance)
        object.__setattr__(self, "leverage", leverage)
        object.__setattr__(self, "maintenance_margin_rate", maintenance)
        object.__setattr__(self, "maker_fee", maker_fee)
        object.__setattr__(self, "taker_fee", taker_fee)

    @property
    def initial_margin_rate(self) -> Decimal:
        return Decimal(1) / self.leverage

    def build_backend_config(self) -> Mapping[str, Any]:
        """构造Binance单向净持仓模拟Venue配置。

        合约中的 ``margin_init`` 和 ``margin_maint`` 已经是最终保证金率，因此使用
        ``StandardMarginModel`` 直接按 ``名义价值 × 保证金率`` 计算。若沿用
        Nautilus默认的 ``LeveragedMarginModel``，它还会额外除以账户杠杆，造成
        保证金被重复折算。
        """

        return GenericVenueProfile(
            profile_id=self.profile_id,
            venue=self.venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(self.starting_balance, USDT)],
            base_currency=USDT,
            default_leverage=self.leverage,
            margin_model=StandardMarginModel(),
            fee_model=MakerTakerFeeModel(),
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
        price_precision: int,
        size_precision: int,
        price_increment: Decimal | int | float | str,
        size_increment: Decimal | int | float | str,
        base_currency: Currency,
        min_quantity: Decimal | int | float | str | None = None,
        min_notional: Decimal | int | float | str | None = None,
    ) -> PerpetualContract:
        """按数据集对应的交易规则创建线性USDT永续合约。"""

        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol不能为空")
        if normalized.endswith("-PERP"):
            raw_symbol = normalized[:-5]
            instrument_symbol = normalized
        else:
            raw_symbol = normalized
            instrument_symbol = f"{normalized}-PERP"
        price_step = _decimal(price_increment)
        size_step = _decimal(size_increment)
        if price_precision < 0 or size_precision < 0:
            raise ValueError("价格和数量精度不能为负数")
        if price_step <= 0 or size_step <= 0:
            raise ValueError("价格和数量最小变动必须大于零")

        instrument_id = InstrumentId.from_str(f"{instrument_symbol}.{self.venue}")
        return PerpetualContract(
            instrument_id=instrument_id,
            raw_symbol=Symbol(raw_symbol),
            underlying=base_currency.code,
            asset_class=AssetClass.CRYPTOCURRENCY,
            base_currency=base_currency,
            quote_currency=USDT,
            settlement_currency=USDT,
            is_inverse=False,
            price_precision=price_precision,
            size_precision=size_precision,
            price_increment=Price.from_str(str(price_step)),
            size_increment=Quantity.from_str(str(size_step)),
            multiplier=Quantity.from_int(1),
            lot_size=None,
            min_quantity=(
                None if min_quantity is None else Quantity.from_str(str(_decimal(min_quantity)))
            ),
            min_notional=(
                None if min_notional is None else Money(_decimal(min_notional), USDT)
            ),
            margin_init=self.initial_margin_rate,
            margin_maint=self.maintenance_margin_rate,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            ts_event=0,
            ts_init=0,
            info={"market_type": "USDT_PERPETUAL", "profile_id": self.profile_id},
        )
