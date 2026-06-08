# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from ultron.ump.core.helper import pd_resample
from ultron.ump.indicator.ma import calc_ma_from_prices, EMACalcType
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin


class FactorSWWaveBuy(FactorBuyID):

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
        super(FactorSWWaveBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},ewm={}'.format(self.__class__.__name__,
                                                    self.ma_xd, self.ewm)

    def _dynamic_calc_xd(self, bar):
        last_kl = self.past_bar_kl(bar=bar, past_bar_cnt=self.resample_max)

        if last_kl.empty:
            return self.ma_xd

        for xd in np.arange(self.resample_min, self.resample_max, 2):
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

        diff_wave = (self.xd_kl.high -
                     self.xd_kl.open) - (1 - self.xd_kl.low / self.xd_kl.open)

        diff_ma = calc_ma_from_prices(prices=diff_wave,
                                      time_period=self.ma_xd,
                                      from_calc=from_calc)
        return diff_ma


class FactorSWWaveBuyL(FactorSWWaveBuy, BuyCallMixin):

    def fit_bar(self, bar):
        diff_ma = super(FactorSWWaveBuyL, self).fit_bar(bar)
        if diff_ma[-1] > 0:
            return self.buy_next()


class FactorSWWaveBuyS(FactorSWWaveBuy, BuyPutMixin):

    def fit_bar(self, bar):
        diff_ma = super(FactorSWWaveBuyS, self).fit_bar(bar)
        if diff_ma[-1] < 0:
            return self.buy_next()
