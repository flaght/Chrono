# -*- encoding:utf-8 -*-
import math
from lumina.factors.sell.fixes import FactorSellXD, ESupportDirection
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices

class FactorWaterFallSell(FactorSellXD):
    def _init_self(self, **kwargs):
        self.p_n1 = kwargs.pop('p_n1', 5)
        
        self.p_n2 = kwargs.pop('p_n2', 10)
 
        self.p_n3 = kwargs.pop('p_n3', 15)

        kwargs['xd'] = max(max(self.p_n3, self.p_n2), self.p_n1) + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorWaterFallSell, self)._init_self(**kwargs)

        self.factor_name = '{}:p_n1={},p_n2={},p_n3={}'.format(self.__class__.__name__,
                                                       self.p_n1,self.p_n2,self.p_n3)
        

    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]
    
    def fit_day(self, today, orders):
        waterfall1 = calc_ma_from_prices(
               self.xd_kl.close, int(self.p_n1), min_periods=1)
          
        waterfall2 = calc_ma_from_prices(
                self.xd_kl.close, int(self.p_n2), min_periods=1)
          
        waterfall3 = calc_ma_from_prices(
                self.xd_kl.close, int(self.p_n3), min_periods=1)
        
        for order in orders:
            if order.expect_direction == 1 and today.close < waterfall2[-1] and waterfall1[-1] < waterfall2[-2]:
                return self.sell_tomorrow(order)
            elif order.expect_direction == 1 and today.close < waterfall3[-1]:
                return self.sell_tomorrow(order)
            elif order.expect_direction == -1 and today.close > waterfall2[-1] and waterfall1[-1] > waterfall2[-2]:
                return self.sell_tomorrow(order)
            elif order.expect_direction == -1 and today.close > waterfall3[-1]:
                return self.sell_tomorrow(order)