# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from ultron.ump.core.helper import pd_resample
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin


class FacotorBaMaBuy(FactorBuyID):

    def _init_self(self, **kwargs):
        self.ma_xd = kwargs.pop('ma_xd', 5)
        if self.ma_xd == -1:
            self.ma_xd = 5
            self.dynamic_xd = True

        self.resample_min = kwargs.pop('resample_min', 5)

        self.resample_max = kwargs.pop('resample_max', 20)

        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        self.bama_threshold = kwargs.pop('bama_threshold', 1.12)

        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1

        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FacotorBaMaBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},ewm={}'.format(self.__class__.__name__,
                                                    self.ma_xd, self.ewm)

    def _dynamic_calc_xd(self, bar):
        last_kl = self.past_bar_kl(bar=bar, past_bar_cnt=self.resample_max)

        if last_kl.empty:
            return self.ma_xd

        for xd in np.arange(self.resample_min, self.resample_max, 2):
            rule = "{}T".format(xd) #分钟重采样
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
        # ## 成交量平均线
        if self.ewm == 1:
            ama_line = pd_ewm_mean(self.xd_kl.volume, span=int(self.ma_xd), min_periods=1)
            bma_line = pd_ewm_mean(self.xd_kl.close, span=int(self.ma_xd), min_periods=1)
        else:
            ama_line = pd_rolling_mean(self.xd_kl.volume, window=int(self.ma_xd), min_periods=1)
            bma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        
        bma_chg = bma_line.pct_change()
        ama_chg = ama_line.pct_change()

        bama = bma_chg * ama_chg

        return bama


class FacotorBaMaBuyL(FacotorBaMaBuy, BuyCallMixin):

    def fit_bar(self, bar):
        bama = super(FacotorBaMaBuyL, self).fit_bar(bar)
        if bama.iloc[-1] > self.bama_threshold:
            return self.buy_next()


class FacotorBaMaBuyS(FacotorBaMaBuy, BuyPutMixin):

    def fit_bar(self, bar):
        bama = super(FacotorBaMaBuyS, self).fit_bar(bar)

        if bama.iloc[-1] < self.bama_threshold:
            return self.buy_next()