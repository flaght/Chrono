# -*- encoding:utf-8 -*-
import math, pdb
import numpy as np
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.indicator.ma import EMACalcType
from lumina.factors.sell.fixes import FactorSellID, ESupportDirection


class FactorTiChaosSell(FactorSellID):

    def _init_self(self, **kwargs):
        self.threshold = kwargs.pop('threshold', 0.12)
        self.ewm = kwargs.pop('ewm', 1)
        kwargs['xd'] = kwargs.pop('ma', 1)

        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorTiChaosSell, self)._init_self(**kwargs)

        self.factor_name = '{}:threshold={},ewm={},ma={}'.format(
            self.__class__.__name__, self.threshold, self.ewm, self.xd)

    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]

    def fit_day(self, today, orders):
        pass

    def fit_bar(self, bar, orders):
        for order in orders:
            if order.expect_direction == 1 and bar.pred < self.threshold:
                self.sell_next(order)
            if order.expect_direction == -1 and bar.pred > self.threshold:
                self.sell_next(order)
