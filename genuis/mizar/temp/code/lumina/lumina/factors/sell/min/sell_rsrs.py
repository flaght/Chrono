# -*- encoding:utf-8 -*-
import math, pdb
import numpy as np
from ultron.ump.core.helper import pd_resample, pd_rolling_corr, pd_ewm_corr, pd_rolling_mean, pd_ewm_mean
from lumina.factors.sell.fixes import FactorSellID, ESupportDirection
from lumina.factors.fixed import rolling_2p


def ols_beta(values, target):
    y = np.nan_to_num(values)
    x = np.nan_to_num(target)
    X = np.hstack((np.ones(
        (x.shape[0], 1)), x))  # add constant X = sm.add_constant(X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
    return beta


class FactorRSRSSell(FactorSellID):

    def _init_self(self, **kwargs):
        self.ma_xd = kwargs.pop('ma_xd', 40)
        self.rsrs_threshold = kwargs.pop('rsrs_threshold', 0.5)
        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorRSRSSell, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},ewm={},rsrs_threshold={}'.format(
            self.__class__.__name__, self.ma_xd, self.ewm, self.rsrs_threshold)

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

        for order in orders:
            if order.expect_direction == 1 and rsrs[-1] < self.rsrs_threshold:
                self.sell_next(order)
            elif order.expect_direction == -1 and rsrs[
                    -1] > self.rsrs_threshold:
                self.sell_next(order)
