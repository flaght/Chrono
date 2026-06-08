# -*- encoding:utf-8 -*-
"""
    买入择时示例因子： 瀑布策略
"""
import math, pdb
import numpy as np
from ultron.ump.core.helper import pd_resample
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.indicator.atr import calc_atr
from lumina.factors.buy.fixes import FactorBuyXD, BuyCallMixin, BuyPutMixin


class FactorWaterFallBuy(FactorBuyXD):
    """示例WaterFall策略，混入BuyCallMixin"""

    def _init_self(self, **kwargs):
        self.p_n1 = kwargs.pop('p_n1', -1)
        self.dynamic_n1 = False
        if self.p_n1 == -1:
            self.p_n1 = 5
            self.dynamic_n1 = True

        self.p_n2 = kwargs.pop('p_n2', -1)
        self.dynamic_n2 = False
        if self.p_n2 == -1:
            self.p_n2 = 10
            self.dynamic_n2 = True


        self.p_n3 = kwargs.pop('p_n3', -1)
        self.dynamic_n3 = False
        if self.p_n3 == -1:
            self.p_n3 = 15
            self.dynamic_n3 = True

        # 动态慢线可设置参数重采样周期最大值，默认50
        self.resample_max = kwargs.pop('resample_max', 20)
        # 动态慢线可设置参数重采样周期最小值，默认10
        self.resample_min = kwargs.pop('resample_min', 5)
        # 动态慢线可设置参数代表慢线的选取阀值，默认0.12
        self.change_threshold = kwargs.pop('change_threshold', 0.12)


        kwargs['xd'] = max(max(self.p_n3, self.p_n2), self.p_n1) + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorWaterFallBuy, self)._init_self(**kwargs)

        self.factor_name = '{}:p_n1={},p_n2={},p_n3={}'.format(self.__class__.__name__,
                                                       self.p_n1,self.p_n2,self.p_n3)
    
    def _dynamic_calc_ma(self, today, ma_xd):
        last_kl = self.past_today_kl(today=today,
                                     past_day_cnt=self.resample_max)
        
        if last_kl.empty:
            return ma_xd
        
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if  benchmark_today.empty:
            return math.ceil(ma_xd * 1)
        
        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_today.iloc[0].key)
        start_key = end_key - 20
        if start_key < 0:
            return math.ceil(ma_xd * 1)
        
        # 使用切片切出从今天开始向前20天的数据
        benchmark_month = benchmark_df.set_index('key').loc[start_key:end_key +
                                                            1].reset_index()
        
        # 通过大盘最近一个月的收盘价格做为参数构造TLine对象
        benchmark_month_line = Line(benchmark_month.close, 'benchmark month line')

        # 计算这个月最少需要几次拟合才能代表走势曲线
        least, _, _, _, _, _  =  benchmark_month_line.create_best_poly()
        if least == 1:
            return math.ceil(ma_xd * 1.05)
        elif least == 2:
            return math.ceil(ma_xd * 1.15)
        elif least == 3:
            return math.ceil(ma_xd * 1.30)
        else:
            return math.ceil(ma_xd * 1.50)
        
    def fit_month(self, today):
        if self.dynamic_n1:
            self.p_n1 = self._dynamic_calc_ma(today, self.p_n1)
        if self.dynamic_n2:
            self.p_n2 = self._dynamic_calc_ma(today, self.p_n2)
        if self.dynamic_n3:
            self.p_n3 = self._dynamic_calc_ma(today, self.p_n3)

        # 动态重新计算后，改变在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:p_n1={},p_n2={},p_n3={}'.format(self.__class__.__name__,
                                                       self.p_n1,self.p_n2,self.p_n3)



class FactorWaterFallBuyL(FactorWaterFallBuy, BuyCallMixin):
     
    def fit_day(self, today):

        waterfall1 = calc_ma_from_prices(
               self.xd_kl.close, int(self.p_n1), min_periods=1)
          
        waterfall2 = calc_ma_from_prices(
                self.xd_kl.close, int(self.p_n2), min_periods=1)
          
        waterfall3 = calc_ma_from_prices(
                self.xd_kl.close, int(self.p_n3), min_periods=1)
        
        if today.close > waterfall1[-1] and waterfall1[-1] > waterfall2[-1] and waterfall2[-1] > waterfall3[-1]:
            return self.buy_tomorrow()
        

class FactorWaterFallBuyS(FactorWaterFallBuy, BuyPutMixin):
    def fit_day(self, today):
        waterfall1 = calc_ma_from_prices(
               self.xd_kl.close, int(self.p_n1), min_periods=1)
          
        waterfall2 = calc_ma_from_prices(
                self.xd_kl.close, int(self.p_n2), min_periods=1)
          
        waterfall3 = calc_ma_from_prices(
                self.xd_kl.close, int(self.p_n3), min_periods=1)
        
        if today.close < waterfall1[-1] and waterfall1[-1] < waterfall2[-1] and waterfall2[-1] < waterfall3[-1]:
            return self.sell_tomorrow()