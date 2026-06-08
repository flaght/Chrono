# -*- encoding:utf-8 -*-
import pdb, math
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.core.helper import  pd_resample
from ultron.ump.indicator.ma import EMACalcType
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin


class FactorChaosBuy(FactorBuyID):

    def _init_self(self, **kwargs):
        self.fast = kwargs.pop('fast', -1)
        self.dynamic_fast = False
        if self.fast == -1:
            self.fast = 3
            self.dynamic_fast = True

        self.slow = kwargs.pop('slow', -1)
        self.dynamic_slow = False
        if self.slow == -1:
            self.slow = 5
            self.dynamic_slow = True

        self.ewm = kwargs.pop('ewm', 1)

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

        self.factor_name = '{}:fast={},slow={},ewm={}'.format(
            self.__class__.__name__, self.fast, self.slow, self.ewm)

    def fit_day(self, bar):
        if self.dynamic_slow:
            self.slow = self._dynamic_calc_slow(bar)

        if self.dynamic_fast:
            self.fast = self._dynamic_calc_fast(bar)

        self.factor_name = '{}:fast={},slow={},ewm={}'.format(
            self.__class__.__name__, self.fast, self.slow, self.ewm)

    #
    def _dynamic_calc_fast(self, bar):
        benchmark_df = self.benchmark.kl_pd
        benchmark_bar = benchmark_df[benchmark_df.ttime == bar.ttime]
        if benchmark_bar.empty:
            return math.ceil(self.slow * 0.25)

        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_bar.iloc[0].key)
        start_key = end_key - self.cycle
        if start_key < 0:
            return math.ceil(self.slow * 0.25)
        # 使用切片切出从今天开始向前20周期的数据
        benchmark_date = benchmark_df.set_index('key').loc[start_key:end_key +
                                                           1].reset_index()

        # 通过大盘最近一个月的收盘价格做为参数构造TLine对象
        benchmark_date_line = Line(benchmark_date.close,
                                   'benchmark month line')
        # 计算这个月最少需要几次拟合才能代表走势曲线
        least, _, _, _, _, _ = benchmark_date_line.create_least_valid_poly()
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

    def _dynamic_calc_slow(self, bar):
        last_kl = self.past_bar_kl(bar=bar, past_bar_cnt=self.resample_max)

        if last_kl.empty:
            return 5

        for slow in np.arange(self.resample_min, self.resample_max, 1):
            rule = "{}T".format(slow)
            change = abs(
                pd_resample(last_kl.close, rule,
                            how='mean').pct_change()).mean()
            if change > self.change_threshold:
                return slow
        # 迭代np.arange(min, max, 1)都不符合就返回max
        return self.resample_max


class FactorChaosBuyL(FactorChaosBuy, BuyCallMixin):

    def fit_bar(self, bar):
        n3 = self.fast + self.slow
        n4 = n3 + self.slow

        hl = (self.xd_kl.high + self.xd_kl.low) / 2

        from_calc = EMACalcType.E_MA_EMA if self.ewm else EMACalcType.E_MA_MA

        Y = calc_ma_from_prices(hl.shift(n3),
                                int(n4),
                                min_periods=1,
                                from_calc=from_calc)

        R = calc_ma_from_prices(hl.shift(self.slow),
                                int(n3),
                                min_periods=1,
                                from_calc=from_calc)

        G = calc_ma_from_prices(hl.shift(self.fast),
                                int(self.slow),
                                min_periods=1,
                                from_calc=from_calc)

        H1 = self.xd_kl.high[:-3]
        h_array = np.where(H1.values == self.xd_kl.high[:-6].max)[-1]
        top_n = (0 + 2) if len(h_array) == 0 else h_array[-1] + 2

        top_line = self.xd_kl.high[:-top_n]

        max_yrg = np.maximum(Y, R, G)

        # 收盘价升破上分形，并且上分形在鳄鱼线上方时，多头开仓
        if bar.close > top_line[-1] and top_line[-1] > max_yrg[
                -1] and bar.high > max_yrg[-1]:
            return self.buy_next()


class FactorChaosBuyS(FactorChaosBuy, BuyPutMixin):

    def fit_bar(self, bar):
        n3 = self.fast + self.slow
        n4 = n3 + self.slow

        hl = (self.xd_kl.high + self.xd_kl.low) / 2

        from_calc = EMACalcType.E_MA_EMA if self.ewm else EMACalcType.E_MA_MA

        Y = calc_ma_from_prices(hl.shift(n3),
                                int(n4),
                                min_periods=1,
                                from_calc=from_calc)

        R = calc_ma_from_prices(hl.shift(self.slow),
                                int(n3),
                                min_periods=1,
                                from_calc=from_calc)

        G = calc_ma_from_prices(hl.shift(self.fast),
                                int(self.slow),
                                min_periods=1,
                                from_calc=from_calc)

        L1 = self.xd_kl.low[:-3]
        l_array = np.where(L1.values == self.xd_kl.low[:-6].min)[-1]
        bottom_n = (0 + 2) if len(l_array) == 0 else l_array[-1] + 2

        bottom_line = self.xd_kl.low[:-bottom_n]

        min_yrg = np.minimum(Y, R, G)

        # 收盘价跌破下分形，并且下分形在鳄鱼线下方时，空头开仓
        if bar.close < bottom_line[-2] and bottom_line[-2] < min_yrg[
                -2] and bar.low < min_yrg[-2]:
            return self.buy_next()
