# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子，基于贪婪布林通道的突破
"""
from lumina.factors.sell.fixes import FactorSellXD, ESupportDirection
from ultron.ump.core.helper import pd_rolling_std, pd_ewm_std, pd_rolling_mean

class FactorDisplaceBollSell(FactorSellXD):
    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：xd: 均线周期，默认不设置，使用自适应动态快线
        """
        self.ma_xd = kwargs.pop('ma_xd', 3)

        self.std_xd = kwargs.pop('std_xd', 12)
        
        self.sdev = kwargs.pop('sdev', 2)

        self.ewm = kwargs.pop('ewm', 1)

        self.disp = kwargs.pop('disp', 12)


        kwargs['xd'] = self.ma_xd + self.disp
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorDisplaceBollSell, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},std_xd={}'.format(self.__class__.__name__,
                                                       self.ma_xd,self.std_xd)
        
    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]
    

    def fit_day(self, today, orders):
        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if len(ma_line) < int(self.std_xd):
            return

        if len(ma_line) > int(self.std_xd):
            if self.ewm == 1:
                band = pd_ewm_std(self.xd_kl.close, span=int(self.std_xd), min_periods=1, adjust=False)
            else:
                band = pd_rolling_std(self.xd_kl.close, window=int(self.std_xd), min_periods=1, center=False) 
        
            dmult = band * self.sdev

            disp_top = ma_line.shift(self.disp) + dmult

            disp_bot = ma_line.shift(self.disp) - dmult

            for order in orders:
                if order.expect_direction == 1 and today.low < disp_bot.iloc[-1]:
                    self.sell_tomorrow(order)
                elif order.expect_direction == -1 and today.high > disp_top.iloc[-1]:
                    self.sell_tomorrow(order)