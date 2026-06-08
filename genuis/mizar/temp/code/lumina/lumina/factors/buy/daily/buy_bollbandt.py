# -*- encoding:utf-8 -*-
import pdb, math
from ultron.ump.technical.line import Line
from ultron.ump.core.helper import pd_rolling_std, pd_ewm_std, pd_rolling_mean, pd_resample
from lumina.factors.buy.fixes import FactorBuyXD, BuyCallMixin, BuyPutMixin



class FactorBollBandtBuy(FactorBuyXD):
    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：xd: 均线周期，默认不设置，使用自适应动态快线
        """
        self.ma_xd = kwargs.pop('ma_xd', -1)
        self.dynamic_ma = False
        if self.ma_xd == -1:
            self.ma_xd = 40
            self.dynamic_ma = True
        
        self.offset = kwargs.pop('offset', 1.25)

        self.ewm = kwargs.pop('ewm', 1)

        self.roc_length = kwargs.pop('roc', 30)

        # 动态可设置参数重采样周期最大值，默认90
        self.resample_max = kwargs.pop('resample_max', 80)
        # 动态可设置参数重采样周期最小值，默认10
        self.resample_min = kwargs.pop('resample_min', 10)
        # 动态可设置参数代表慢线的选取阀值，默认0.12
        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        kwargs['xd'] = self.ma_xd + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorBollBandtBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},ewm={}'.format(self.__class__.__name__,
                                                       self.ma_xd,self.ewm)
        
    def _dynamic_calc_ma(self, today):
        
        last_kl = self.past_today_kl(today=today,
                                     past_day_cnt=self.resample_max)
        
        if last_kl.empty:
            return 60
         # 迭代np.arange(min, max, 5)都不符合就返回max
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

        self.factor_name = '{}:ma={},ewm={}'.format(self.__class__.__name__,
                                                       self.ma_xd,self.ewm)
        
        
class FactorBollBandtBuyL(FactorBollBandtBuy, BuyCallMixin):
    def fit_day(self, today):
        """价格上破布林通道上轨，开多单"""
        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close, span=int(self.ma_xd), min_periods=1, adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close, window=int(self.ma_xd), min_periods=1, center=False) 

        roc_price = self.xd_kl.close.diff(self.roc_length)
        if today.high > (ma_line + self.offset * band).iloc[-1] and roc_price.iloc[-1] > 0:
            return self.buy_tomorrow()
        
        

class FactorBollBandtBuyS(FactorBollBandtBuy, BuyPutMixin):
    def fit_day(self, today):
        """价格上破布林通道下轨，开空单"""
        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close, span=int(self.ma_xd), min_periods=1, adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close, window=int(self.ma_xd), min_periods=1, center=False) 

        roc_price = self.xd_kl.close.diff(self.roc_length)
        if today.low < (ma_line + self.offset * band).iloc[-1] and roc_price.iloc[-1] < 0:
            return self.buy_tomorrow()
        