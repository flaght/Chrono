"""公共原生合约构造，不依赖数据准备层或示例策略。"""
from __future__ import annotations

from datetime import datetime, time, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo


def precision(tick):
    tick = Decimal(str(tick))
    if not tick.is_finite() or tick <= 0:
        raise ValueError("Invalid price increment")
    return max(0, -tick.normalize().as_tuple().exponent)


def day_ns(day, clock="00:00", timezone_name="Asia/Shanghai"):
    stamp = datetime.combine(day, time.fromisoformat(clock), ZoneInfo(timezone_name))
    delta = stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000


def make_profile_future(profile, *, symbol, product, tick, multiplier, listed,
                        last_trade, margin_init, margin_maint):
    from market.basic.base import InstrumentMeta
    tick, multiplier = Decimal(str(tick)), Decimal(str(multiplier))
    if not multiplier.is_finite() or multiplier <= 0:
        raise ValueError("Invalid contract multiplier")
    places = precision(tick)
    contract = profile.make_instrument(symbol, underlying=product.lower(),
        price_precision=places, price_increment=tick, multiplier=multiplier,
        # 沿用商品期货模拟配置的合约生命周期时间口径。
        activation_ns=day_ns(listed, timezone_name="UTC"),
        expiration_ns=day_ns(last_trade, timezone_name="UTC") + 86_400_000_000_000,
        margin_init=margin_init, margin_maint=margin_maint)
    meta = InstrumentMeta(contract.id, price_precision=places, size_precision=0,
        price_increment=tick, multiplier=multiplier, currency="CNY", exchange=str(profile.venue))
    return contract, meta, multiplier


def make_option(info, margin_init, margin_maint, index_code, *, kind="C", profile_id="OPTION_VEGA_BASIC"):
    """构造真实期权合约；默认认购，认沽调用方须显式传 kind='P'。"""
    if kind not in ("C", "P"):
        raise ValueError("Option kind must be C or P")
    from bomber.model import Price, Quantity, Symbol
    from bomber.model.currencies import CNY
    from bomber.model.enums import AssetClass, OptionKind
    from bomber.model.identifiers import InstrumentId
    from bomber.model.instruments import OptionContract
    places = precision(info.tick)
    return OptionContract(instrument_id=InstrumentId.from_str(info.symbol + ".CFFEX"),
        raw_symbol=Symbol(info.symbol), asset_class=AssetClass.INDEX, currency=CNY,
        price_precision=places, price_increment=Price.from_str(format(info.tick.normalize(), "f")),
        multiplier=Quantity.from_str(format(info.multiplier.normalize(), "f")),
        lot_size=Quantity.from_int(1), underlying=index_code,
        option_kind=OptionKind.CALL if kind == "C" else OptionKind.PUT,
        strike_price=Price.from_str(f"{info.strike:.{places}f}"),
        activation_ns=day_ns(info.list_day), expiration_ns=day_ns(info.last_day, "15:00"),
        margin_init=margin_init, margin_maint=margin_maint, ts_event=0, ts_init=0, exchange="CFFEX",
        info={"profile": profile_id, "expiry_policy": "PRE_EXPIRY_FLAT"})


def make_future(symbol, row, margin_init, margin_maint):
    from bomber.model import Price, Quantity, Symbol
    from bomber.model.currencies import CNY
    from bomber.model.enums import AssetClass
    from bomber.model.identifiers import InstrumentId
    from bomber.model.instruments import FuturesContract
    tick = row["minChgPriceNum"]
    return FuturesContract(instrument_id=InstrumentId.from_str(symbol + ".CFFEX"),
        raw_symbol=Symbol(symbol), asset_class=AssetClass.INDEX, currency=CNY,
        price_precision=precision(tick), price_increment=Price.from_str(format(tick.normalize(), "f")),
        multiplier=Quantity.from_str(format(row["contMultNum"].normalize(), "f")),
        lot_size=Quantity.from_int(1), underlying=str(row["code"]).upper(),
        activation_ns=day_ns(row["listDate"]), expiration_ns=day_ns(row["lastTradeDate"], "15:00"),
        margin_init=margin_init, margin_maint=margin_maint, ts_event=0, ts_init=0, exchange="CFFEX")


def instrument_meta(contract):
    from market.basic.base import InstrumentMeta
    return InstrumentMeta(contract.id, price_precision=contract.price_precision, size_precision=0,
        price_increment=Decimal(str(contract.price_increment)), multiplier=Decimal(str(contract.multiplier)),
        currency="CNY", exchange="CFFEX")
