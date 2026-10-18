# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子，KingKeltner策略
"""
from lumina.factors.sell.fixes import FactorSellID, ESupportDirection
from ultron.ump.indicator.ma import calc_ma_from_prices, EMACalcType


class FactorKingKeltnerSell(FactorSellID):
    """示例卖出KingKeltner因子"""

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：xd: 均线周期，默认不设置，使用自适应动态快线
        """
        self.ma_xd = kwargs.pop('ma_xd', 40)
        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorKingKeltnerSell, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.sell_type_extra = '{}:ma={},xd={},ewm={}'.format(
            self.__class__.__name__, self.ma_xd, self.xd, self.ewm)

    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]

    def fit_day(self, today, orders):
        pass

    def fit_bar(self, bar, orders):
        """
            双均线卖出择时因子：
            call方向：快线下穿慢线形成死叉，做为卖出信号 多头
            put方向： 快线上穿慢线做为卖出信号 空头
        """
        ### 均线不够，无法计算或没有订单
        if len(orders) == 0 or len(self.xd_kl) < self.xd:
            return

        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA
        # 计算均线
        ma_line = calc_ma_from_prices(
            (self.xd_kl.close + self.xd_kl.high + self.xd_kl.low) / 3,
            int(self.ma_xd),
            min_periods=1,
            from_calc=from_calc)

        for order in orders:
            if order.expect_direction == 1 \
                and bar.low <= ma_line[-1]:
                # 多头 价格下破三价均线，平多单
                self.sell_next(order)
            elif order.expect_direction == -1 \
                and bar.high >= ma_line[-1]:
                # 空头 价格上穿三价均线，平空单
                self.sell_next(order)
