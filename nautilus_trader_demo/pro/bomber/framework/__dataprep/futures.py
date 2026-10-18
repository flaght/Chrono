"""为现有期货回测调用方提供轻量兼容组装。"""
from decimal import Decimal

from .metadata import contract_rows, future_spec, selected_products, venue, venue_of, symbol_of


def make_instrument(row, profile, margin_init=Decimal("0.10"), margin_maint=Decimal("0.08")):
    from bomber.framework.trader.instrument_factory import make_profile_future
    spec = future_spec(row)
    if spec.venue != str(profile.venue):
        raise ValueError(f"{spec.symbol}: profile venue differs from metadata")
    return make_profile_future(profile, symbol=symbol_of(row), product=spec.product,
        tick=spec.tick, multiplier=spec.multiplier, listed=spec.listed, last_trade=spec.last_trade,
        margin_init=margin_init, margin_maint=margin_maint)


def instrument(basic, profile, product, symbol, margin_init=Decimal("0.10"), margin_maint=Decimal("0.08")):
    row = contract_rows(basic, {symbol})[symbol]
    if str(row["code"]).strip().upper() != product.upper():
        raise ValueError(f"{symbol}: product differs from metadata")
    row = row.copy()
    row["symbol"] = symbol
    return make_instrument(row, profile, margin_init, margin_maint)
