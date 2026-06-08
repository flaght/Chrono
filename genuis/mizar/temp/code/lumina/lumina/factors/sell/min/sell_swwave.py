# -*- encoding:utf-8 -*-
import pdb
from lumina.factors.sell.fixes import FactorSellID, ESupportDirection
from ultron.ump.indicator.ma import calc_ma_from_prices, EMACalcType


class FactorSWWaveSell(FactorSellID):

    def _init_self(self, **kwargs):
        self.ma_xd = kwargs.pop('ma_xd', 5)
        self.ewm = kwargs.pop('ewm', 1)
        kwargs['xd'] = self.ma_xd + 1
        super(FactorSWWaveSell, self)._init_self(**kwargs)
        self.factor_name = '{}:ma={},ewm={}'.format(self.__class__.__name__,
                                                    self.ma_xd, self.ewm)

    def support_direction(self):
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]

    def fit_day(self, today, orders):
        pass

    def fit_bar(self, bar, orders):
        ### 均线不够，无法计算或没有订单
        if len(orders) == 0 or len(self.xd_kl) < self.xd:
            return
        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA

        diff_wave = (self.xd_kl.high -
                     self.xd_kl.open) - (1 - self.xd_kl.low / self.xd_kl.open)

        diff_ma = calc_ma_from_prices(prices=diff_wave,
                                      time_period=self.ma_xd,
                                      from_calc=from_calc)
        for order in orders:
            if order.expect_direction == 1 and diff_ma[-1] < 0:
                return self.sell_next(order)
            elif order.expect_direction == -1 and diff_ma[-1] > 0:
                return self.sell_next(order)
