# -*- encoding:utf-8 -*-
import numpy as np
from lumina.factors.sell.fixes import FactorSellID, ESupportDirection
from ultron.ump.indicator.ma import calc_ma_from_prices, EMACalcType
from lumina.factors.fixed import rolling_1p


def slopes(values):
    y = np.nan_to_num(values)
    row_indices = np.arange(y.shape[0])
    x = np.tile(row_indices[:, np.newaxis], (1, y.shape[1]))
    X = np.hstack((np.ones(
        (x.shape[0], 1)), x))  # add constant X = sm.add_constant(X)
    slopes = np.linalg.lstsq(X, y, rcond=None)[0]
    slope = np.median(slopes, axis=0)
    intercepts = y - slope * x
    intercept = np.median(intercepts, axis=0)
    return intercept + slope * (x[-1] + 1)


class FactorICUSell(FactorSellID):

    def _init_self(self, **kwargs):
        self.ma_xd = kwargs.pop('ma_xd', 40)
        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorICUSell, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},ewm={}'.format(self.__class__.__name__,
                                                    self.ma_xd, self.ewm)

    def support_direction(self):
        """支持的方向，因子支持两个方向"""
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

        icu = rolling_1p(value=self.xd_kl.close,
                         window=self.ma_xd,
                         func1=slopes,
                         name='icu')
        icu_ma = calc_ma_from_prices(icu,
                                     int(self.ma_xd),
                                     min_periods=1,
                                     from_calc=from_calc)

        for order in orders:
            if order.expect_direction == 1 and self.xd_kl.close[-1] < icu_ma[
                    -1] and self.xd_kl.close[-2] > icu_ma[-2]:
                self.sell_next(order)
            elif order.expect_direction == -1 and self.xd_kl.close[
                    -1] > icu_ma[-1] and self.xd_kl.close[-2] < icu_ma[-2]:
                self.sell_next(order)
