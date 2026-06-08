# -*- encoding:utf-8 -*-
import pdb, math
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.indicator.rsi import _calc_rsi_from_pd
from ultron.ump.core.helper import pd_rolling_min, pd_rolling_max, pd_resample
from ultron.ump.indicator.ma import EMACalcType
from lumina.factors.buy.fixes import FactorBuyXD, BuyCallMixin, BuyPutMixin


class FactorChaosBuy(FactorBuyXD):
    def _init_self(self, **kwargs):
        self.fast = kwargs.pop('fast', -1)
        self.dynamic_n1 = False
        if self.fast == -1:
            self.fast = 3
            self.dynamic_fast = True

        self.slow = kwargs.pop('slow', -1)
        self.dynamic_slow = False
        if self.slow == -1:
            self.slow = 5
            self.dynamic_slow = True

        kwargs['xd'] = (self.slow + self.fast + 1) * 2

         # 动态慢线可设置参数重采样周期最大值，默认15
        self.resample_max = kwargs.pop('resample_max', 15)
        # 动态慢线可设置参数重采样周期最小值，默认3
        self.resample_min = kwargs.pop('resample_min', 3)
        # 动态慢线可设置参数代表慢线的选取阀值，默认0.12
        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorChaosBuy, self)._init_self(**kwargs)

        self.least = 0

        self.factor_name = '{}:fast={},slow={},least={}'.format(self.__class__.__name__,
                                                       self.fast,self.slow, self.least)
        

    # 当级周期
    def _dynamic_calc_fast(self, today):
        benchmark_df = self.benchmark.kl_pd
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            return math.ceil(self.slow * 0.25)

        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_today.iloc[0].key)
        start_key = end_key - 20
        if start_key < 0:
            return math.ceil(self.slow * 0.25)
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
            return math.ceil(self.slow * 0.2)
        elif least == 2:
            # 二次拟合可以表达：
            return math.ceil(self.slow * 0.3)
        elif least == 3:
            # 三次拟合可以表达：
            return math.ceil(self.slow * 0.4)
        else:
            # 四次及以上拟合可以表达：
            return math.ceil(self.slow * 0.5)

    def _dynamic_calc_slow(self, today):
        last_kl = self.past_today_kl(today=today,
                                     past_day_cnt=self.resample_max)

        if last_kl.empty:
            return 5

        for slow in np.arange(self.resample_min, self.resample_max, 1):
            rule = "{}D".format(slow)
            change = abs(
                pd_resample(last_kl.close, rule,
                            how='mean').pct_change()).mean()
            if change > self.change_threshold:
                return slow
        # 迭代np.arange(min, max, 1)都不符合就返回max
        return self.resample_max
    
    def fit_month(self, today):
        if self.dynamic_slow:
            self.slow = self._dynamic_calc_slow(today)
        
        if self.dynamic_fast:
            self.fast = self._dynamic_calc_fast(today)

        self.factor_name = '{}:fast={},slow={},least={}'.format(self.__class__.__name__,
                                                       self.fast,self.slow,self.least)



    

class FactorChaosBuyL(FactorChaosBuy, BuyCallMixin):
    def fit_day(self, today):

        n3 = self.fast + self.slow
        n4 = n3 + self.slow

        hl = (self.xd_kl.high + self.xd_kl.low) / 2


        Y = calc_ma_from_prices(
               hl.shift(n3), int(n4), min_periods=1, from_calc=EMACalcType.E_MA_EMA)
        
        R = calc_ma_from_prices(
               hl.shift(self.slow), int(n3), min_periods=1, from_calc=EMACalcType.E_MA_EMA)
        
        G = calc_ma_from_prices(
               hl.shift(self.fast), int(self.slow), min_periods=1, from_calc=EMACalcType.E_MA_EMA)
        
        H1 = self.xd_kl.high[:-3]
        h_array = np.where(H1.values == self.xd_kl.high[:-6].max)[-1]
        top_n = (0 + 2) if len(h_array) == 0 else h_array[-1] + 2 

        top_line = self.xd_kl.high[:-top_n]

        max_yrg = np.maximum(Y, R, G)

        
        # 收盘价升破上分形，并且上分形在鳄鱼线上方时，多头开仓
        if today.close > top_line[-1] and top_line[-1] > max_yrg[-1] and today.high > max_yrg[-1]:
            return self.buy_tomorrow()

        

class FactorChaosBuyS(FactorChaosBuy, BuyCallMixin):
    def fit_day(self, today):

        n3 = self.fast + self.slow
        n4 = n3 + self.slow

        hl = (self.xd_kl.high + self.xd_kl.low) / 2


        Y = calc_ma_from_prices(
               hl.shift(n3), int(n4), min_periods=1, from_calc=EMACalcType.E_MA_EMA)
        
        R = calc_ma_from_prices(
               hl.shift(self.slow), int(n3), min_periods=1, from_calc=EMACalcType.E_MA_EMA)
        
        G = calc_ma_from_prices(
               hl.shift(self.fast), int(self.slow), min_periods=1, from_calc=EMACalcType.E_MA_EMA)
        

        L1 = self.xd_kl.low[:-3]
        l_array = np.where(L1.values == self.xd_kl.low[:-6].min)[-1]
        bottom_n = (0 + 2) if len(l_array) == 0 else l_array[-1] + 2

        bottom_line = self.xd_kl.low[:-bottom_n]

        min_yrg = np.minimum(Y, R, G)

        
        # 收盘价跌破下分形，并且下分形在鳄鱼线下方时，空头开仓
        if today.close < bottom_line[-2] and bottom_line[-2] < min_yrg[-2] and today.low < min_yrg[-2]:
            return self.buy_tomorrow()



