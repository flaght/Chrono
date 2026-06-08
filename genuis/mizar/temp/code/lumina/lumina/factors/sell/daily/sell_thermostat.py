# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.indicator.atr import calc_atr
from ultron.ump.core.helper import pd_rolling_min, pd_rolling_max, pd_rolling_std, pd_ewm_std, pd_rolling_mean
from lumina.factors.sell.fixes import FactorSellXD, ESupportDirection

class FactorThermostatSell(FactorSellXD):
    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：xd:
        """
        self.mv_xd = kwargs.pop('mv_xd', 30)
        self.swing_trend_switch = kwargs.pop('swing_trend_switch', 20) # 潮汐指数小于此值为震荡市，否则为趋势市
        self.swing_price1 = kwargs.pop('swing_price1', 0.5) # 震荡市中的价格
        self.swing_price2 = kwargs.pop('swing_price2', 0.75) # 震荡市中的价格
        self.atr_xd = kwargs.pop('atr_xd', 10)

        # 布林通道参数
        self.ma_xd = kwargs.pop('ma_xd', 50)
        self.sdev = kwargs.pop('sdev', 2)

        self.trend_xd = kwargs.pop('trend_xd', 50) # 趋势市时进场单的出场均线参数

        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1

        super(FactorThermostatSell, self)._init_self(**kwargs)

        self.factor_name = '{}:mv_xd={},swing_trend_switch={},ma_xd={},ewm={}'.format(
            self.__class__.__name__,self.ma_xd,self.swing_trend_switch,self.ma_xd,self.ewm)
        
    
    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [
            ESupportDirection.DIRECTION_CAll.value,
            ESupportDirection.DIRECTION_PUT.value
        ]
    

    def fit_day(self, today, orders):
        if len(self.xd_kl) < self.ma_xd:
            return 
        high_price = pd_rolling_max(self.xd_kl.high, window=self.mv_xd)
        low_price = pd_rolling_min(self.xd_kl.low, window=self.mv_xd )
        pre_price = self.xd_kl.close.shift(self.mv_xd - 1)

        #  潮汐指数区分震荡市与趋势市
        cmi_val = np.abs(
            (self.xd_kl.close - pre_price) / (
               high_price -  low_price
            )
        ) * 100

        # 震荡市中收盘价大于关键价格为宜卖市， 否则为宜买市
        buy_easier_day = False
        sell_easier_day = False

        # 计算关键价格
        key_of_day  = (self.xd_kl.high + self.xd_kl.low + self.xd_kl.close) / 3

        if today.close <= key_of_day[-1]:
            buy_easier_day = True
        else:
            sell_easier_day = True

        # 计算震荡市进场价格
        atr_line = calc_atr(self.xd_kl.high, 
                            self.xd_kl.low, 
                            self.xd_kl.close, int(self.atr_xd))
        
        if buy_easier_day:
            swing_buy_pt = today.open + self.swing_price1 * atr_line[-1]
            swing_sell_pt = today.open + self.swing_price2 * atr_line[-1]
        elif sell_easier_day:
            swing_buy_pt = today.open + self.swing_price2 * atr_line[-1]
            swing_sell_pt = today.open + self.swing_price1 * atr_line[-1]

        swing_buy_pt = max(swing_buy_pt, key_of_day[-1])
        swing_sell_pt = min(swing_sell_pt, key_of_day[-1])

        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close, span=int(self.ma_xd), min_periods=1, adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close, window=int(self.ma_xd), min_periods=1, center=False) 

        up_band = ma_line + self.sdev * band
        down_band = ma_line - self.sdev * band

        trend_buy_pt = up_band
        trend_sell_pt = down_band

        swing_prot_stop = 3 * atr_line
        trend_port_stop = pd_rolling_mean(self.xd_kl.close, window=int(self.trend_xd), min_periods=1)
        for order in orders:
            # 震荡市 平仓
            if order.expect_direction == 1 and today.low < swing_prot_stop[-1] and cmi_val[-1] < self.swing_trend_switch:
                return self.sell_tomorrow(order)
            elif order.expect_direction == -1 and today.high > swing_prot_stop[-1] and cmi_val[-1] < self.swing_trend_switch:
                return self.sell_tomorrow(order)
            # 趋势市 平仓
            elif order.expect_direction == 1 and today.low < max(trend_sell_pt[-1], trend_port_stop[-1]) and cmi_val[-1] >= self.swing_trend_switch:
                return self.sell_tomorrow(order)
            elif order.expect_direction == -1 and today.high > min(trend_buy_pt[-1], trend_port_stop[-1]) and cmi_val[-1] >= self.swing_trend_switch:
                return self.sell_tomorrow(order)