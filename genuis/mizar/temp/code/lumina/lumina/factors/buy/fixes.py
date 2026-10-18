try:
    from ultron.ump.factor.buy.base import FactorBuyBase, FactorBuyXD, FactorBuyTD, FactorBuyID, BuyCallMixin, BuyPutMixin
 
except ImportError:
    from lumina.factors.buy.base import FactorBuyBase, FactorBuyXD, FactorBuyTD, BuyCallMixin, BuyPutMixin