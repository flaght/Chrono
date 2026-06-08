# -*- encoding:utf-8 -*-
import pdb
import numpy as np
import pandas as pd
from ultron.ump.core.helper import pd_resample
from ultron.ump.indicator.ma import calc_ma_from_prices, EMACalcType
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin
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


class FactorICUBuy(FactorBuyID):

    def _init_self(self, **kwargs):
        self.ma_xd = kwargs.pop('ma_xd', 5)
        if self.ma_xd == -1:
            self.ma_xd = 5
            self.dynamic_xd = True

        self.resample_min = kwargs.pop('resample_max', 5)

        self.resample_max = kwargs.pop('resample_max', 20)

        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1

        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorICUBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},ewm={}'.format(self.__class__.__name__,
                                                    self.ma_xd, self.ewm)

    def _dynamic_calc_xd(self, bar):
        last_kl = self.past_bar_kl(bar=bar, past_bar_cnt=self.resample_max)

        if last_kl.empty:
            return self.ma_xd

        for xd in np.arange(self.resample_min, self.resample_max, 5):
            rule = "{}T".format(xd)
            change = abs(
                pd_resample(last_kl.close, rule,
                            how='mean').pct_change()).mean()
            if change > self.change_threshold:
                return xd
        return self.ma_xd

    ## 通过天数据刷新min 拟合参数
    def fit_day(self, bar):
        self.ma_slow = self._dynamic_calc_xd(bar)
        self.factor_name = '{}:ma={},ewm={}'.format(self.__class__.__name__,
                                                    self.ma_xd, self.ewm)

    def fit_bar(self, bar):
        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA

        icu = rolling_1p(value=self.xd_kl.close,
                         window=self.ma_xd,
                         func1=slopes,
                         name='icu')
        icu_ma = calc_ma_from_prices(icu,
                                     int(self.ma_xd),
                                     min_periods=1,
                                     from_calc=from_calc)
        return icu_ma


class FactorICUBuyL(FactorICUBuy, BuyCallMixin):

    def fit_bar(self, bar):
        icu_ma = super(FactorICUBuyL, self).fit_bar(bar)
        if self.xd_kl.close[-1] > icu_ma[-1] and self.xd_kl.close[-2] < icu_ma[
                -2]:
            return self.buy_next()


class FactorICUBuyS(FactorICUBuy, BuyPutMixin):

    def fit_bar(self, bar):
        icu_ma = super(FactorICUBuyS, self).fit_bar(bar)
        if self.xd_kl.close[-1] < icu_ma[-1] and self.xd_kl.close[-2] > icu_ma[
                -2]:
            return self.buy_next()
