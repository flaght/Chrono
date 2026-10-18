# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：n倍atr(止盈止损)择时卖出策略
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import pdb
from lumina.factors.sell.fixes import FactorSellIB, ESupportDirection


class FactorAtrNStop(FactorSellIB):
    """示例n倍atr(止盈止损)因子"""

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数stop_loss_n: 止损的atr倍数
            kwargs中可选参数stop_win_n: 止盈的atr倍数
        """

        if 'stop_loss_n' in kwargs:
            # 设置止损的atr倍数
            self.stop_loss_n = kwargs['stop_loss_n']
            # 在输出生成的orders_pd中及可视化等等显示的名字
            self.sell_type_extra_loss = '{}:stop_loss={}'.format(self.__class__.__name__, self.stop_loss_n)

        if 'stop_win_n' in kwargs:
            # 设置止盈的atr倍数
            self.stop_win_n = kwargs['stop_win_n']
            # 在输出生成的orders_pd中及可视化等等显示的名字
            self.sell_type_extra_win = '{}:stop_win={}'.format(self.__class__.__name__, self.stop_win_n)

    def support_direction(self):
        """n倍atr(止盈止损)因子支持两个方向"""
        return [ESupportDirection.DIRECTION_CAll.value, ESupportDirection.DIRECTION_PUT.value]

    def fit_day(self, today, orders):
        pass

    def fit_bar(self, bar, orders):
        for order in orders:
            profit = (bar.close - order.buy_price) * order.expect_direction
            stop_base = bar.atr21 + bar.atr14

            if hasattr(self, 'stop_win_n') and profit > 0 and profit > self.stop_win_n * stop_base:
                # 满足止盈条件卖出股票, 即收益(profit) > n倍atr
                self.sell_type_extra = self.sell_type_extra_win
                # 由于使用了当天的close价格，所以明天才能卖出
                self.sell_next(order)

            if hasattr(self, 'stop_loss_n') and profit < 0 and profit < -self.stop_loss_n * stop_base:
                # 满足止损条件卖出股票, 即收益(profit) < -n倍atr
                self.sell_type_extra = self.sell_type_extra_loss
                order.fit_sell_order(self.bar_ind, self)
                # 由于使用了当天的close价格，所以明天才能卖出
                self.sell_next(order)