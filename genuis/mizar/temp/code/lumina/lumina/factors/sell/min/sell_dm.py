# -*- encoding:utf-8 -*-
from lumina.factors.sell.fixes import FactorSellID, ESupportDirection
from ultron.ump.indicator.ma import calc_ma_from_prices, EMACalcType


class FactorDoubleMaSell(FactorSellID):
    """示例卖出双均线择时因子"""

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：fast: 均线快线周期，默认不设置，使用5
            kwargs中可选参数：slow: 均线慢线周期，默认不设置，使用60
        """
        # TODO 重构与买入因子重复代码抽取
        # 均线快线周期，默认使用5天均线
        self.ma_fast = kwargs.pop('fast', 5)
        # 均线慢线周期，默认使用60天均线
        self.ma_slow = kwargs.pop('slow', 60)

        if self.ma_fast >= self.ma_slow:
            # 慢线周期必须大于快线
            raise ValueError('ma_fast >= self.ma_slow !')

        self.ewm = kwargs.pop('ewm', 1)
        # xd周期数据需要比ma_slow大一天，这样计算ma就可以拿到今天和昨天两天的ma，用来判断金叉，死叉
        kwargs['xd'] = self.ma_slow + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorDoubleMaSell, self)._init_self(**kwargs)

    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]
    
    def fit_day(self, bar, orders):
        pass

    def fit_bar(self, bar, orders):
        """
            双均线卖出择时因子：
            call方向：快线下穿慢线形成死叉，做为卖出信号
            put方向： 快线上穿慢线做为卖出信号
        """
        if len(orders) == 0 or len(self.xd_kl) < self.xd:
            return
        
        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA
        # 计算快线
        fast_line = calc_ma_from_prices(self.xd_kl.close,
                                        self.ma_fast,
                                        min_periods=1,
                                        from_calc=from_calc)
        # 计算慢线
        slow_line = calc_ma_from_prices(self.xd_kl.close,
                                        self.ma_slow,
                                        min_periods=1,
                                        from_calc=from_calc)

        if len(fast_line) >= 2 and len(slow_line) >= 2:
            # 今天的快线值
            fast_current = fast_line[-1]
            # 昨天的快线值
            fast_last = fast_line[-2]
            # 今天的慢线值
            slow_current = slow_line[-1]
            # 昨天的慢线值
            slow_last = slow_line[-2]

            for order in orders:
                if order.expect_direction == 1 \
                        and fast_last >= slow_last and fast_current < slow_current:
                    # call方向：快线下穿慢线线形成死叉，做为卖出信号
                    self.sell_next(order)
                elif order.expect_direction == -1 \
                        and slow_last >= fast_last and fast_current > slow_current:
                    # put方向：快线上穿慢线做为卖出信号
                    self.sell_next(order)
