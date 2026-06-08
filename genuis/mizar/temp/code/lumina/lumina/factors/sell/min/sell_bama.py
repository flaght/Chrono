# -*- encoding:utf-8 -*-
from lumina.factors.sell.fixes import FactorSellID, ESupportDirection
from ultron.ump.core.helper import pd_rolling_mean, pd_ewm_mean


class FactorBaMaSell(FactorSellID):

    def _init_self(self, **kwargs):
        self.ma_xd = kwargs.pop('ma_xd', 5)
        self.bama_threshold = kwargs.pop('bama_threshold', 1.12)
        self.ewm = kwargs.pop('ewm', 1)
        kwargs['xd'] = self.ma_xd + 1
        super(FactorBaMaSell, self)._init_self(**kwargs)
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
        if self.ewm == 1:
            ama_line = pd_ewm_mean(self.xd_kl.volume,
                                   span=int(self.ma_xd),
                                   min_periods=1)
            bma_line = pd_ewm_mean(self.xd_kl.close,
                                   span=int(self.ma_xd),
                                   min_periods=1)
        else:
            ama_line = pd_rolling_mean(self.xd_kl.volume,
                                       window=int(self.ma_xd),
                                       min_periods=1)
            bma_line = pd_rolling_mean(self.xd_kl.close,
                                       window=int(self.ma_xd),
                                       min_periods=1)

        bma_chg = bma_line.pct_change()
        ama_chg = ama_line.pct_change()

        bama = bma_chg * ama_chg

        for order in orders:
            if order.expect_direction == 1 and bama.iloc[
                    -1] < self.bama_threshold:
                return self.sell_next(order)
            elif order.expect_direction == -1 and bama.iloc[
                    -1] > self.bama_threshold:
                return self.sell_next(order)
