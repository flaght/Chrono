# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：恒温器
策略说明:
	通过计算市场的潮汐指数，把市场划分为震荡和趋势两种走势；震荡市中采用开盘区间突破进场；趋势市中采用布林通道突破进场。
系统要素:
		1、潮汐指数
		2、关键价格
		3、布林通道
		4、真实波幅
		5、出场均线
入场条件:
		1、震荡市中采用开盘区间突破进场
		2、趋势市中采用布林通道突破进场
出场条件:
		1、震荡市时进场单的出场为反手信号和ATR保护性止损
		2、趋势市时进场单的出场为反手信号和均线出场
"""

import pdb, math
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.indicator.atr import calc_atr
from ultron.ump.core.helper import pd_rolling_min, pd_rolling_max, pd_rolling_std, pd_ewm_std, pd_rolling_mean, pd_resample
from lumina.factors.buy.fixes import FactorBuyXD, BuyCallMixin, BuyPutMixin

class FactorThermostatBuy(FactorBuyXD):
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
        self.ma_xd = kwargs.pop('ma_xd', -1)
        self.dynamic_ma = False
        if self.ma_xd == -1:
            self.ma_xd = 50
            self.dynamic_ma = True

        self.sdev = kwargs.pop('sdev', 2)

        self.trend_xd = kwargs.pop('trend_xd', 50) # 趋势市时进场单的出场均线参数

        self.resample_max = kwargs.pop('resample_max', 100)

        self.resample_min = kwargs.pop('resample_min', 10)

        self.change_threshold = kwargs.pop('change_threshold', 0.12)


        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1

        self.least = 0

        super(FactorThermostatBuy, self)._init_self(**kwargs)

        self.factor_name = '{}:mv_xd={},swing_trend={},ma_xd={},ewm={},least={}'.format(
            self.__class__.__name__,self.mv_xd,self.swing_trend_switch,
            self.ma_xd,self.ewm,self.least)


    def _dynamic_calc_ma(self, today):
        last_kl = self.past_today_kl(today=today, past_day_cnt=self.resample_max)

        if last_kl.empty:
            return self.ma_xd
        
        ma_value = self.resample_max

        for ma in range(self.resample_min, self.resample_max, 5):
            rule = "{}D".format(ma)
            change = abs(
                pd_resample(last_kl.close, rule,
                            how='mean').pct_change()).mean()
            
            if change > self.change_threshold:
                """
                    返回第一个大于change_threshold,
                    change_threshold默认为0.12，以周期突破的策略一般需要在0.08以上，0.12是为快线留出套利空间
                """
                ma_value = ma

        benchmark_df = self.benchmark.kl_pd
        
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            # 默认值为慢线的0.15
            return math.ceil(ma_value * (1 + 0.15))
        
        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_today.iloc[0].key)
        start_key = end_key - 20
        if start_key < 0:
            # 默认值为慢线的0.15
            return math.ceil(self.ma_xd * 0.15)
        # 使用切片切出从今天开始向前20天的数据
        benchmark_month = benchmark_df.set_index('key').loc[start_key:end_key +
                                                            1].reset_index()
        
        # 通过大盘最近一个月的收盘价格做为参数构造TLine对象
        benchmark_month_line = Line(benchmark_month.close,
                                    'benchmark month line')
        # 计算这个月最少需要几次拟合才能代表走势曲线
        least, _, _, _, _, _ = benchmark_month_line.create_least_valid_poly()

        self.least = least
        if least == 1:
            # 一次拟合可以表达：
            return math.ceil(ma_value * 1.05)
        elif least == 2:
            # 二次拟合可以表达：
            return math.ceil(ma_value * 1.15)
        elif least == 3:
            # 三次拟合可以表达：
            return math.ceil(ma_value * 1.3)
        else:
            # 四次及以上拟合可以表达：
            return math.ceil(ma_value * 1.5)
        
    def fit_month(self, today):
        if self.dynamic_ma:
            self.ma_xd = self._dynamic_calc_ma(today)

        self.factor_name = '{}:mv_xd={},swing_trend={},ma_xd={},ewm={},least={}'.format(
            self.__class__.__name__,self.mv_xd,self.swing_trend_switch,
            self.ma_xd,self.ewm,self.least)


class FactorThermostatBuyL(FactorThermostatBuy, BuyCallMixin):

    def fit_day(self, today):
        
        high_price = pd_rolling_max(self.xd_kl.high, window=self.mv_xd)
        low_price = pd_rolling_min(self.xd_kl.low, window=self.mv_xd )
        pre_price = self.xd_kl.close.shift(self.mv_xd - 1)

        #  潮汐指数区分震荡市与趋势市
        cmi_val = np.abs(
            (self.xd_kl.close - pre_price) / (
               high_price -  low_price
            )
        ) * 100


        # 趋势市价格
        trend_lok_buy = calc_ma_from_prices(self.xd_kl.low,int(3),min_periods=1)

        # 计算关键价格
        key_of_day  = (self.xd_kl.high + self.xd_kl.low + self.xd_kl.close) / 3

        # 震荡市中收盘价大于关键价格为宜卖市， 否则为宜买市
        buy_easier_day = False

        if today.close <= key_of_day[-1]:
            buy_easier_day = True

        # 计算震荡市进场价格
        atr_line = calc_atr(self.xd_kl.high, 
                            self.xd_kl.low, 
                            self.xd_kl.close, int(self.atr_xd))
        
        if buy_easier_day:
            swing_buy_pt = today.open + self.swing_price1 * atr_line[-1]
        else:
            swing_buy_pt = today.open + self.swing_price2 * atr_line[-1]

        swing_buy_pt = max(swing_buy_pt, trend_lok_buy[-1])

        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close, span=int(self.ma_xd), min_periods=1, adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close, window=int(self.ma_xd), min_periods=1, center=False) 

        up_band = ma_line + self.sdev * band
        trend_buy_pt = up_band

        # 震荡市
        if cmi_val[-1] < self.swing_trend_switch and today.high >= swing_buy_pt:
            return self.buy_tomorrow()
        # 趋势市
        elif cmi_val[-1] >= self.swing_trend_switch and today.high >= trend_buy_pt[-1]:
            return self.buy_tomorrow()


class FactorThermostatBuyS(FactorThermostatBuy, BuyPutMixin): 
    def fit_day(self, today):
        high_price = pd_rolling_max(self.xd_kl.high, window=30)
        low_price = pd_rolling_min(self.xd_kl.low, window=30)
        pre_price = self.xd_kl.close.shift(29)

        #  潮汐指数区分震荡市与趋势市
        cmi_val = np.abs(
            (self.xd_kl.close - pre_price) / (
               high_price -  low_price
            )
        ) * 100

        trend_lok_sell = calc_ma_from_prices(self.xd_kl.high,int(3),min_periods=1)

        # 计算关键价格
        key_of_day  = (self.xd_kl.high + self.xd_kl.low + self.xd_kl.close) / 3


        # 震荡市中收盘价大于关键价格为宜卖市， 否则为宜买市
        buy_easier_day = False

        if today.close <= key_of_day[-1]:
            buy_easier_day = True
        

        # 计算震荡市进场价格
        atr_line = calc_atr(self.xd_kl.high, 
                            self.xd_kl.low, 
                            self.xd_kl.close, int(self.atr_xd))

        if buy_easier_day:
            swing_sell_pt = today.open - self.swing_price2 * atr_line[-1]
        else:
            swing_sell_pt = today.open - self.swing_price1 * atr_line[-1]
        
        swing_sell_pt = min(swing_sell_pt, trend_lok_sell[-1])

        # 计算趋势市的进场价格
        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close, span=int(self.ma_xd), min_periods=1, adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close, window=int(self.ma_xd), min_periods=1, center=False) 
        
        down_band = ma_line - self.sdev * band

        trend_sell_pt = down_band

        if cmi_val[-1] < self.swing_trend_switch and today.low <= swing_sell_pt[-1]:
            return self.buy_tomorrow()
        elif cmi_val[-1] >= self.swing_trend_switch and today.low <= trend_sell_pt[-1]:
            return self.buy_tomorrow()
