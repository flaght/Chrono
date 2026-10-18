# -*- encoding:utf-8 -*-
from lumina.factors.sell.fixes import FactorSellID, ESupportDirection
from ultron.ump.core.helper import pd_rolling_min, pd_rolling_max


class FactorGhostTraderSell(FactorSellID):

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：fast: 均线快线周期，默认不设置，使用5
            kwargs中可选参数：slow: 均线慢线周期，默认不设置，使用60
        """
        # 唐奇安通道默认参数
        self.tc_period = kwargs.pop('tc_period', 20)

        kwargs['xd'] = self.tc_period + 5
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorGhostTraderSell, self)._init_self(**kwargs)

    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]

    def fit_day(self, bar, orders):
        pass

    def fit_bar(self, bar, orders):
        # 计算唐奇安通道
        if len(self.xd_kl) < self.xd and len(orders) == 0:
            return

        hi_band = pd_rolling_max(self.xd_kl.high,
                                 window=self.tc_period,
                                 min_periods=1)
        lo_band = pd_rolling_min(self.xd_kl.low,
                                 window=self.tc_period,
                                 min_periods=1)

        for order in orders:
            # 下破唐奇安通道下轨，平多单
            if order.expect_direction == 1 and bar.low < lo_band[-2]:
                return self.sell_next(order)
            elif order.expect_direction == -1 and bar.high > hi_band[-2]:
                return self.sell_next(order)
