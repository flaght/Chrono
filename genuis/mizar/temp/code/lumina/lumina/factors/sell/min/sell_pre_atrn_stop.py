# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：单日最大跌幅n倍atr止损
    做为单边止损因子使用，作为风险控制保护因子
"""

from lumina.factors.sell.fixes import FactorSellIB, ESupportDirection
"""外部可通过如：g_default_pre_atr_n = 2.5来修改默认值"""
g_default_pre_atr_n = 1.5


class FactorPreAtrNStop(FactorSellIB):
    """示例单日最大跌幅n倍atr(止损)风险控制因子"""

    def _init_self(self, **kwargs):
        """kwargs中可选参数pre_atr_n: 单日最大跌幅止损的atr倍数"""

        self.pre_atr_n = g_default_pre_atr_n
        if 'pre_atr_n' in kwargs:
            # 设置下跌止损倍数
            self.pre_atr_n = kwargs['pre_atr_n']
            self.sell_type_extra = '{}:pre_atr={}'.format(
                self.__class__.__name__, self.pre_atr_n)

    def support_direction(self):
        """单日最大跌幅n倍atr(止损)因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]

    def fit_day(self, bar, orders):
        pass

    def fit_bar(self, bar, orders):
        for order in orders:
            if (bar.pre_close - bar.close
                ) * order.expect_direction > bar.atr21 * self.pre_atr_n:
                # 只要今天的收盘价格比昨天收盘价格差大于一个差值就止损卖出, 亦可以使用其它计算差值方式
                self.sell_next(order)
