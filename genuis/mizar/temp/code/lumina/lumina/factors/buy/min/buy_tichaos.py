### 主要针对模型的阈值进行调整# -*- encoding:utf-8 -*-
import pdb, math
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.core.helper import pd_resample
from ultron.ump.indicator.ma import EMACalcType
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin


class FactorTiChaosBuy(FactorBuyID):

    def _init_self(self, **kwargs):
        ###  阈值
        self.threshold = kwargs.pop('threshold', 0.12)
        self.ewm = kwargs.pop('ewm', 1)
        kwargs['xd'] = kwargs.pop('ma', 1)
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorTiChaosBuy, self)._init_self(**kwargs)
        self.factor_name = '{}:threshold={},ewm={},ma={}'.format(
            self.__class__.__name__, self.threshold, self.ewm, self.xd)


class FactorTiChaosBuyL(FactorTiChaosBuy, BuyCallMixin):
    def fit_day(self, bar):
        pass

    def fit_bar(self, bar):
        if bar.pred > self.threshold:
            return self.buy_next()
