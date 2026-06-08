# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：基于平移的boll通道突破

系统要素:
		1、平移的boll通道
入场条件:
		1、关键价格突破通道上轨，则开多仓
		2、关键价格突破通道下轨，则开空仓
出场条件:
		1、关键价格突破通道上轨，则平空仓
		2、关键价格突破通道下轨，则平多仓
"""
import pdb, math
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.core.helper import pd_resample, pd_rolling_std, pd_ewm_std, pd_rolling_mean
from lumina.factors.buy.fixes import FactorBuyXD, BuyCallMixin, BuyPutMixin

class FactorDisplaceBollBuy(FactorBuyXD):
    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：xd: 均线周期，默认不设置，使用自适应动态快线
        """
        self.ma_xd = kwargs.pop('ma_xd', -1)
        self.dynamic_ma = False
        if self.ma_xd == -1:
            self.ma_xd = 20
            self.dynamic_ma = True

        self.std_xd = kwargs.pop('std_xd', 12)
        
        self.sdev = kwargs.pop('sdev', 2)

        self.ewm = kwargs.pop('ewm', 1)

        self.disp = kwargs.pop('disp', 12)


        # 动态可设置参数重采样周期最大值，默认90
        self.resample_max = kwargs.pop('resample_max', 80)
        # 动态可设置参数重采样周期最小值，默认10
        self.resample_min = kwargs.pop('resample_min', 10)
        # 动态可设置参数代表慢线的选取阀值，默认0.12
        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        kwargs['xd'] = self.ma_xd + self.disp
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorDisplaceBollBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},std_xd={}'.format(self.__class__.__name__,
                                                       self.ma_xd,self.std_xd)
        
    
    def _dynamic_calc_ma(self, today):
        """
            动态决策值，规则如下：
            1. 切片最近一段时间的金融时间序列，对金融时间序列进行变换周期重新采样，
            对重新采样的结果进行pct_change处理，对pct_change序列取abs绝对值，
            对pct_change绝对值序列取平均，即算出重新采样的周期内的平均变化幅度，
            上述的变换周期由10， 15，20，30....进行迭代, 直到计算出第一个重新
            采样的周期内的平均变化幅度 > 0.12的周期做为的取值,

            2.根据大盘最近一个月走势使用：
            一次拟合可以表达：ma ＝ ma * 1.05
            二次拟合可以表达：ma ＝ ma * 1.15
            三次拟合可以表达：ma ＝ ma * 1.3
            四次及以上拟合可以表达：ma ＝ ma * 1.5
        """
        last_kl = self.past_today_kl(today=today,
                                     past_day_cnt=self.resample_max)
        
        if last_kl.empty:
            return 20
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

        self.factor_name = '{}:ma={}, std_xd={}'.format(
            self.__class__.__name__,self.ma_xd,self.std_xd)


class FactorDisplaceBollBuyL(FactorDisplaceBollBuy, BuyCallMixin):

    def fit_day(self, today):
        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close, span=int(self.std_xd), min_periods=1, adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close, window=int(self.std_xd), min_periods=1, center=False) 
        
        dmult = band * self.sdev

        disp_top = ma_line.shift(self.disp) + dmult

        if today.high > disp_top.iloc[-1]:
            return self.buy_tomorrow()
    

        
class FactorDisplaceBollBuyS(FactorDisplaceBollBuy, BuyPutMixin):
    def fit_day(self, today):
        ma_line = pd_rolling_mean(self.xd_kl.close, window=int(self.ma_xd), min_periods=1)
        if self.ewm == 1:
            band = pd_ewm_std(self.xd_kl.close, span=int(self.std_xd), min_periods=1, adjust=False)
        else:
            band = pd_rolling_std(self.xd_kl.close, window=int(self.std_xd), min_periods=1, center=False) 
        
        dmult = band * self.sdev

        disp_bottom = ma_line.shift(self.disp) - dmult

        if today.low < disp_bottom:
            return self.buy_tomorrow() 