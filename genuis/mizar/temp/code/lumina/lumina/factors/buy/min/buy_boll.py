# -*- encoding:utf-8 -*-

import pdb, math
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_std, pd_ewm_std, pd_rolling_mean, pd_ewm_mean, pd_resample
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin


class FactorBollBuy(FactorBuyID):

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：xd: 均线周期，默认不设置，使用自适应动态快线
        """
        self.ma_xd = kwargs.pop('ma_xd', 40)
        if self.ma_xd == -1:
            self.ma_xd = 5
            self.dynamic_xd = True

        self.resample_min = kwargs.pop('resample_min', 3)

        self.resample_max = kwargs.pop('resample_max', 10)

        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        self.ewm = kwargs.pop('ewm', 1)

        self.roc_length = kwargs.pop('roc', 30)

        self.offset = kwargs.pop('offset', 2)

        kwargs['xd'] = self.ma_xd + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorBollBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},ewm={},change_threshold={},roc={},offset={}'.format(
            self.__class__.__name__, self.ma_xd, self.ewm,
            self.change_threshold, self.roc_length, self.offset)

    def _dynamic_calc_xd(self, bar):
        last_kl = self.past_bar_kl(bar=bar, past_bar_cnt=self.resample_max)

        if last_kl.empty:
            return self.ma_xd

        for xd in np.arange(self.resample_min, self.resample_max, 2):
            rule = "{}T".format(xd)  #分钟重采样
            change = abs(
                pd_resample(last_kl.close, rule,
                            how='mean').pct_change()).mean()
            if change > self.change_threshold:
                return xd
        return self.ma_xd

        ## 通过天数据刷新min 拟合参数
    def fit_day(self, bar):
        self.ma_slow = self._dynamic_calc_xd(bar)
        self.factor_name = '{}:ma={},ewm={},change_threshold={},roc={},offset={}'.format(
            self.__class__.__name__, self.ma_xd, self.ewm,
            self.change_threshold, self.roc_length, self.offset)

    def fit_bar(self, bar):
        """价格上破布林通道上轨，开多单"""
        if self.ewm == 1:
            ma_line = pd_ewm_mean(self.xd_kl.close,
                                  span=int(self.ma_xd),
                                  min_periods=1)
        else:
            ma_line = pd_rolling_mean(self.xd_kl.close,
                                      window=int(self.ma_xd),
                                      min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close,
                              span=int(self.ma_xd),
                              min_periods=1,
                              adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close,
                                  window=int(self.ma_xd),
                                  min_periods=1,
                                  center=False)

        roc_price = self.xd_kl.close.diff(self.roc_length)

        return ma_line, band, roc_price


class FactorBollBuyL(FactorBollBuy, BuyCallMixin):

    def fit_bar(self, bar):
        ma_line, band, roc_price = super(FactorBollBuyL, self).fit_bar(bar)
        if bar.high > (ma_line +
                       self.offset * band).iloc[-1] and roc_price.iloc[-1] > 0:
            return self.buy_next()


class FactorBollBuyS(FactorBollBuy, BuyPutMixin):

    def fit_bar(self, bar):
        ma_line, band, roc_price = super(FactorBollBuyS, self).fit_bar(bar)
        if bar.low < (ma_line -
                      self.offset * band).iloc[-1] and roc_price.iloc[-1] < 0:
            return self.buy_next()
