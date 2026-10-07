HAS_CYTHON_CORE = False

try:
    from bomber.model.data import Bar, BarSpecification, BarType, QuoteTick, TradeTick
    from bomber.model.enums import AggressorSide, BarAggregation, PriceType
    from bomber.model.identifiers import InstrumentId, Symbol, Venue, TradeId
    from bomber.model.objects import Price, Quantity
    HAS_CYTHON_CORE = True
except ImportError:
    try:
        from nautilus_trader.model.data import Bar, BarSpecification, BarType, QuoteTick, TradeTick
        from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType
        from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, TradeId
        from nautilus_trader.model.objects import Price, Quantity
        HAS_CYTHON_CORE = True
    except ImportError:
        HAS_CYTHON_CORE = False