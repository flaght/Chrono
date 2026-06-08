# -*- encoding:utf-8 -*-
try:
    from ultron.ump.trade.ml_feature import BuyFeatureMixin, SellFeatureMixin, FeatureBase
except ImportError:
    from lumina.features.base import BuyFeatureMixin, SellFeatureMixin, FeatureBase