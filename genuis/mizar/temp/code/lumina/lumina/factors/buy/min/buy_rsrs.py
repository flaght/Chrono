# -*- encoding:utf-8 -*-
import math
import numpy as np
from ultron.ump.core.helper import pd_resample, pd_rolling_corr, pd_ewm_corr, pd_rolling_mean, pd_ewm_mean
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin
from lumina.factors.fixed import rolling_2p


def ols_beta(values, target):
    y = np.nan_to_num(values)
    x = np.nan_to_num(target)
    X = np.hstack((np.ones(
        (x.shape[0], 1)), x))  # add constant X = sm.add_constant(X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
    return beta


class FactorRSRSBuy(FactorBuyID):

    def _init_self(self, **kwargs):
        self.ma_xd = kwargs.pop('ma_xd', 5)
        if self.ma_xd == -1:
            self.ma_xd = 5
            self.dynamic_xd = True

        self.resample_min = kwargs.pop('resample_max', 5)

        self.resample_max = kwargs.pop('resample_max', 20)

        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        self.rsrs_threshold = kwargs.pop('rsrs_threshold', 0.5)

        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1

        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorRSRSBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},ewm={},change_threshold={},rsrs_threshold={}'.format(
            self.__class__.__name__, self.ma_xd, self.ewm,
            self.change_threshold, self.rsrs_threshold)

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
        self.factor_name = '{}:ma={},ewm={},change_threshold={},rsrs_threshold={}'.format(
            self.__class__.__name__, self.ma_xd, self.ewm,
            self.change_threshold, self.rsrs_threshold)

    def fit_bar(self, bar):
        beta1 = rolling_2p(value1=self.xd_kl.low,
                           value2=self.xd_kl.high,
                           window=self.ma_xd,
                           func1=ols_beta,
                           name='beta1')

        if self.ewm == 1:
            corr1 = pd_ewm_corr(self.xd_kl.low,
                                self.xd_kl.high,
                                span=self.ma_xd,
                                min_periods=1)
            mean_rsrs = pd_ewm_mean(beta1, span=self.ma_xd, min_periods=1)
        else:
            corr1 = pd_rolling_corr(self.xd_kl.low,
                                    self.xd_kl.high,
                                    window=self.ma_xd,
                                    min_periods=1)
            mean_rsrs = pd_rolling_mean(beta1,
                                        window=self.ma_xd,
                                        min_periods=1)

        ### 右偏修正标准分RSRS = 标准RSRS * 相关系数 * RSRS
        rsrs = mean_rsrs * corr1 * beta1
        return rsrs


class FactorRSRSBuyL(FactorRSRSBuy, BuyPutMixin):

    def fit_bar(self, bar):
        rsrs = super(FactorRSRSBuyL, self).fit_bar(bar)
        if rsrs[-1] > self.rsrs_threshold:
            return self.buy_next()


class FactorRSRSBuyS(FactorRSRSBuy, BuyPutMixin):

    def fit_bar(self, bar):
        rsrs = super(FactorRSRSBuyS, self).fit_bar(bar)
        if rsrs[-1] < self.rsrs_threshold:
            return self.buy_next()
