# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子，基于贪婪布林通道的突破
"""
from lumina.factors.sell.fixes import FactorSellXD, ESupportDirection
from ultron.ump.core.helper import pd_rolling_std, pd_ewm_std, pd_rolling_mean

class FactorBollBandtSell(FactorSellXD):
    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：xd: 均线周期，默认不设置，使用自适应动态快线
        """
        self.ma_xd = kwargs.pop('ma_xd', 40)

        self.offset = kwargs.pop('offset', 1.25)

        self.ewm = kwargs.pop('ewm', 1)

        self.roc_length = kwargs.pop('roc', 30)

        kwargs['xd'] = self.ma_xd + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorBollBandtSell, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},xd={}'.format(self.__class__.__name__,
                                                       self.ma_xd,self.xd)
        
    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]
    
    def fit_day(self, today, orders):
        """
            贪婪布林卖出择时因子：
            call方向：均线低于布林通道上轨，并且价格下破自适应出场均线，平多单
            put方向： 出场均线高于布林通道下轨，并且价格上破自适应出场均线，平空单
            自适应出场: 
        """
        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close, span=int(self.ma_xd), min_periods=1, adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close, window=int(self.ma_xd), min_periods=1, center=False) 

        roc_price = self.xd_kl.close.diff(self.roc_length)
        for order in orders:
            if order.expect_direction == 1 \
                and today.low < (ma_line + self.offset * band).iloc[-1] and roc_price.iloc[-1] < 0:
                return self.sell_tomorrow(order)
            elif order.expect_direction == -1 \
                and today.high > (ma_line - self.offset * band).iloc[-1] and roc_price.iloc[-1] > 0:
                return self.sell_tomorrow(order)



    
