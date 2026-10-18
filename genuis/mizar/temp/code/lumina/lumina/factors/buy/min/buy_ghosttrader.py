# -*- encoding:utf-8 -*-
import pdb, math
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.indicator.rsi import _calc_rsi_from_pd
from ultron.ump.core.helper import  pd_resample
from ultron.ump.indicator.ma import EMACalcType
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin


class FactorGhostTraderBuy(FactorBuyID):

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：
        """
        # 均线短期周期，默认使用9天均线
        self.ma_fast = kwargs.pop('fast', -1)
        self.dynamic_fast = False
        if self.ma_fast == -1:
            self.ma_fast = 9
            self.dynamic_fast = True

        # 均线长期周期，默认使用19天均线
        self.ma_slow = kwargs.pop('slow', -1)
        self.dynamic_slow = False
        if self.ma_slow == -1:
            self.ma_slow = 19
            self.dynamic_slow = True

        self.cycle = 200 if 'cycle' in kwargs else kwargs.pop('cycle', 200)

        # RSI 默认参数
        self.rsi_period = kwargs.pop('rsi_period', 14)

        # 超买值
        self.over_bought = kwargs.pop('over_bought', 30)

        # 超卖值
        self.over_sold = kwargs.pop('over_sold', 70)

        # 唐奇安通道默认参数
        self.tc_period = kwargs.pop('tc_period', 20)

        # 动态慢线可设置参数重采样周期最大值，默认50
        self.resample_max = kwargs.pop('resample_max', 20)
        # 动态慢线可设置参数重采样周期最小值，默认10
        self.resample_min = kwargs.pop('resample_min', 5)
        # 动态慢线可设置参数代表慢线的选取阀值，默认0.12
        self.change_threshold = kwargs.pop('change_threshold', 0.08)

        self.ewm = kwargs.pop('ewm', 1)

        if self.ma_fast >= self.ma_slow:
            # 慢线周期必须大于快线
            raise ValueError('ma_fast >= self.ma_slow !')

        # xd周期数据需要比ma_slow大一天，这样计算ma就可以拿到今天和昨天两天的ma，用来判断金叉，死叉
        kwargs['xd'] = self.ma_slow + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorGhostTraderBuy, self)._init_self(**kwargs)

        self.least = 0
        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:fast={},slow={},least={}'.format(
            self.__class__.__name__, self.ma_fast, self.ma_slow, self.least)

    def _dynamic_calc_fast(self, bar):
        # 策略中拥有self.benchmark，即交易基准对象，Benchmark实例对象，benchmark.kl_pd即对应的市场大盘走势
        benchmark_df = self.benchmark.kl_pd

        # 今天的大盘行情
        benchmark_bar = benchmark_df[benchmark_df.ttime == bar.ttime]
        if benchmark_bar.empty:
            # 默认值为慢线的0.15
            return math.ceil(self.ma_slow * 0.3)

        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_bar.iloc[0].key)
        start_key = end_key - self.cycle
        if start_key < 0:
            # 默认值为慢线的0.15
            return math.ceil(self.ma_slow * 0.3)

        # 使用切片切出从今天开始向前cycle天的数据
        benchmark_date = benchmark_df.set_index('key').loc[start_key:end_key +
                                                           1].reset_index()

        # 通过大盘最近一个月的收盘价格做为参数构造TLine对象
        benchmark_date_line = Line(benchmark_date.close, 'benchmark date line')

        # 计算这个月最少需要几次拟合才能代表走势曲线
        least, _, _, _, _, _ = benchmark_date_line.create_least_valid_poly()
        self.least = least
        if least == 1:
            # 一次拟合可以表达：
            return math.ceil(self.ma_slow * 0.2)
        elif least == 2:
            # 二次拟合可以表达：
            return math.ceil(self.ma_slow * 0.3)
        elif least == 3:
            # 三次拟合可以表达：
            return math.ceil(self.ma_slow * 0.6)
        else:
            # 四次及以上拟合可以表达：fast＝slow * 0.5 eg: slow=60->fast=60*0.5=30
            return math.ceil(self.ma_slow * 0.8)

    def _dynamic_calc_slow(self, bar):
        last_kl = self.past_bar_kl(bar=bar, past_bar_cnt=self.resample_max)
        if last_kl.empty:
            return self.resample_max

        for slow in np.arange(self.resample_min, self.resample_max, 3):
            rule = "{}T".format(slow)
            change = abs(
                pd_resample(last_kl.close, rule,
                            how='mean').pct_change()).mean()
            if change > self.change_threshold:
                return slow

        return self.resample_max

    def fit_day(self, bar):
        # fit_month即在回测策略中每天执行一次的方法
        if self.dynamic_slow:
            # 一定要先动态算ma_slow，因为动态计算fast依赖slow
            self.ma_slow = self._dynamic_calc_slow(bar)
        if self.dynamic_fast:
            # 动态计算快线
            self.ma_fast = self._dynamic_calc_fast(bar)
        # 动态重新计算后，改变在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:fast={},slow={},least={}'.format(
            self.__class__.__name__, self.ma_fast, self.ma_slow, self.least)


class FactorGhostTraderBuyL(FactorGhostTraderBuy, BuyCallMixin):

    def fit_bar(self, bar):
        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA

        # 计算快线
        fast_line = calc_ma_from_prices(self.xd_kl.close,
                                        int(self.ma_fast),
                                        min_periods=1,
                                        from_calc=from_calc)
        # 计算慢线
        slow_line = calc_ma_from_prices(self.xd_kl.close,
                                        int(self.ma_slow),
                                        min_periods=1,
                                        from_calc=from_calc)

        # 计算RSI
        rsi = _calc_rsi_from_pd(self.xd_kl.close, self.rsi_period)

        # 计算唐奇安通道
        #hi_band = pd_rolling_max(self.xd_kl.high, window=self.tc_period)
        #lo_band = pd_rolling_min(self.xd_kl.low, window=self.tc_period)

        # 模拟交易产生一次亏损、短期均线在长期均线之上、RSI低于超买值、创新高，则开多单
        if fast_line[-1] > slow_line[-1] and rsi[-1] < self.over_bought \
            and bar.high > self.xd_kl.high[-2]:
            return self.buy_next()


class FactorGhostTraderBuyS(FactorGhostTraderBuy, BuyPutMixin):

    def fit_bar(self, bar):

        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA
        # 计算快线
        fast_line = calc_ma_from_prices(self.xd_kl.close,
                                        int(self.ma_fast),
                                        min_periods=1,
                                        from_calc=from_calc)
        # 计算慢线
        slow_line = calc_ma_from_prices(self.xd_kl.close,
                                        int(self.ma_slow),
                                        min_periods=1,
                                        from_calc=from_calc)

        # 计算RSI
        rsi = _calc_rsi_from_pd(self.xd_kl.close, self.rsi_period)

        # 计算唐奇安通道
        #hi_band = pd_rolling_max(self.xd_kl.high, window=self.tc_period)
        #lo_band = pd_rolling_min(self.xd_kl.low, window=self.tc_period)

        # 短期均线在长期均线之下、RSI高于超卖值、创新低，则开空单
        if fast_line[-1] < slow_line[-1] and rsi[-1] > self.over_sold \
            and bar.low < self.xd_kl.low[-2]:
            return self.buy_next()
