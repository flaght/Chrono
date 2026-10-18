# -*- encoding:utf-8 -*-
import math,pdb
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.indicator.ma import EMACalcType
from lumina.factors.sell.fixes import FactorSellXD, ESupportDirection
from ultron.ump.core.helper import pd_rolling_min, pd_rolling_max, pd_resample



class FactorChaosSell(FactorSellXD):
    def _init_self(self, **kwargs):
        self.fast = kwargs.pop('fast', 3)

        self.slow = kwargs.pop('slow', 5)

        kwargs['xd'] = (self.slow + self.fast + 1) * 2


        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorChaosSell, self)._init_self(**kwargs)


        self.factor_name = '{}:fast={},slow={}'.format(self.__class__.__name__,
                                                       self.fast,self.slow)

    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]
    
    def fit_day(self, today, orders):
        n3 = self.fast + self.slow
        n4 = n3 + self.slow

        hl = (self.xd_kl.high + self.xd_kl.low) / 2

        Y = calc_ma_from_prices(
               hl.shift(n3), int(n4), min_periods=1, from_calc=EMACalcType.E_MA_EMA)
        

        for order in orders:
            if order.expect_direction == 1 and today.close < Y[-1]:
                self.sell_tomorrow(order)
            if order.expect_direction == -1 and today.close > Y[-1]:
                self.sell_tomorrow(order)