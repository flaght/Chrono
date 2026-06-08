# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：幽灵交易者

系统要素:
		1、两条指数平均线
		2、RSI指标
		3、唐奇安通道
入场条件:
        1、模拟交易产生一次亏损、短期均线在长期均线之上、RSI低于超买值、创新高，则开多单
		2、模拟交易产生一次亏损、短期均线在长期均线之下、RSI高于超卖值、创新低，则开空单

出场条件:
        1、持有多单时小于唐奇安通道下轨，平多单
		2、持有空单时大于唐奇安通道上轨，平空单

"""
import pdb
from lumina.factors.sell.fixes import FactorSellXD, ESupportDirection
from ultron.ump.core.helper import pd_rolling_min, pd_rolling_max, pd_resample


class FactorGhostTraderSell(FactorSellXD):

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
    

    def fit_day(self, today, orders):
        # 计算唐奇安通道
        if len(self.xd_kl) < self.xd:
            return None
        hi_band = pd_rolling_max(self.xd_kl.high, window=self.tc_period)
        lo_band = pd_rolling_min(self.xd_kl.low, window=self.tc_period)

        for order in orders:
            # 下破唐奇安通道下轨，平多单
            if order.expect_direction == 1 and today.low < lo_band[-2]:
                return self.sell_tomorrow(order)
            elif order.expect_direction == -1 and today.high > hi_band[-2]:
                return self.sell_tomorrow(order)