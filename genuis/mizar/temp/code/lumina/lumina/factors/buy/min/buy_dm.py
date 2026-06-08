# -*- encoding:utf-8 -*-
import math
import numpy as np
from ultron.ump.core.helper import pd_resample
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices, EMACalcType
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin


class FactorDoubleMaBuy(FactorBuyID, BuyCallMixin):

    def _init_self(self, **kwargs):
        # 均线快线周期，默认使用5天均线
        self.ma_fast = kwargs.pop('fast', -1)
        self.dynamic_fast = False
        if self.ma_fast == -1:
            self.ma_fast = 5
            self.dynamic_fast = True

        # 均线慢线周期，默认使用60天均线
        self.ma_slow = kwargs.pop('slow', -1)
        self.dynamic_slow = False
        if self.ma_slow == -1:
            self.ma_slow = 60
            self.dynamic_slow = True
        # 动态慢线可设置参数重采样周期最大值，默认90
        self.resample_max = kwargs.pop('resample_max', 100)
        # 动态慢线可设置参数重采样周期最小值，默认10
        self.resample_min = kwargs.pop('resample_min', 10)
        # 动态慢线可设置参数代表慢线的选取阀值，默认0.12
        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        self.ewm = kwargs.pop('ewm', 1)

        if self.ma_fast >= self.ma_slow:
            # 慢线周期必须大于快线
            raise ValueError('ma_fast >= self.ma_slow !')

        # xd周期数据需要比ma_slow大一天，这样计算ma就可以拿到今天和昨天两天的ma，用来判断金叉，死叉
        kwargs['xd'] = self.ma_slow + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorDoubleMaBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:fast={},slow={}'.format(self.__class__.__name__,
                                                       self.ma_fast,
                                                       self.ma_slow)

    def _dynamic_calc_fast(self, today):
        # 策略中拥有self.benchmark，即交易基准对象，AbuBenchmark实例对象，benchmark.kl_pd即对应的市场大盘走势
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_bar = benchmark_df[benchmark_df.ttime == today.ttime]
        if benchmark_bar.empty:
            # 默认值为慢线的0.15
            return math.ceil(self.ma_slow * 0.15)

        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_bar.iloc[0].key)
        start_key = end_key - 20
        if start_key < 0:
            # 默认值为慢线的0.15
            return math.ceil(self.ma_slow * 0.15)
        # 使用切片切出从今天开始向前20天的数据
        benchmark_date = benchmark_df.set_index('key').loc[start_key:end_key +
                                                           1].reset_index()

        # 通过大盘最近一个月的收盘价格做为参数构造TLine对象
        benchmark_date_line = Line(benchmark_date.close, 'benchmark date line')
        # 计算这个月最少需要几次拟合才能代表走势曲线
        least, _, _, _, _, _ = benchmark_date_line.create_least_valid_poly()
        if least == 1:
            # 一次拟合可以表达：fast＝slow * 0.05 eg: slow=60->fast=60*0.05=3
            return math.ceil(self.ma_slow * 0.05)
        elif least == 2:
            # 二次拟合可以表达：fast＝slow * 0.15 eg: slow=60->fast=60*0.15=9
            return math.ceil(self.ma_slow * 0.15)
        elif least == 3:
            # 三次拟合可以表达：fast＝slow * 0.3 eg: slow=60->fast=60*0.3=18
            return math.ceil(self.ma_slow * 0.3)
        else:
            # 四次及以上拟合可以表达：fast＝slow * 0.5 eg: slow=60->fast=60*0.5=30
            return math.ceil(self.ma_slow * 0.5)

    def _dynamic_calc_slow(self, bar):
        last_kl = self.past_bar_kl(bar=bar, past_bar_cnt=self.resample_max)

        if last_kl.empty:
            return 60

        for slow in np.arange(self.resample_min, self.resample_max, 5):
            rule = "{}T".format(slow)
            change = abs(
                pd_resample(last_kl.close, rule,
                            how='mean').pct_change()).mean()
            if change > self.change_threshold:
                return slow
        # 迭代np.arange(min, max, 5)都不符合就返回max
        return self.resample_max

    def fit_day(self, bar):
        # fit_month即在回测策略中每一个月执行一次的方法
        if self.dynamic_slow:
            # 一定要先动态算ma_slow，因为动态计算fast依赖slow
            self.ma_slow = self._dynamic_calc_slow(bar)
        if self.dynamic_fast:
            # 动态计算快线
            self.ma_fast = self._dynamic_calc_fast(bar)
        # 动态重新计算后，改变在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:fast={},slow={}'.format(self.__class__.__name__,
                                                       self.ma_fast,
                                                       self.ma_slow)


class FactorDoubleMaBuyL(FactorDoubleMaBuy, BuyCallMixin):

    def fit_bar(self, bar):
        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA
        """双均线买入择时因子，信号快线上穿慢行形成金叉做为买入信号"""
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

        if len(fast_line) >= 2 and len(slow_line) >= 2:
            # 今天的快线值
            fast_current = fast_line[-1]
            # 昨天的快线值
            fast_last = fast_line[-2]
            # 今天的慢线值
            slow_current = slow_line[-1]
            # 昨天的慢线值
            slow_last = slow_line[-2]

            if slow_last >= fast_last and fast_current > slow_current:
                # 快线上穿慢线, 形成买入金叉，使用了今天收盘价格，明天买入
                return self.buy_next()


class FactorDoubleMaBuyS(FactorDoubleMaBuy, BuyPutMixin):

    def fit_bar(self, bar):
        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA
        """双均线买入择时因子，信号快线下穿慢行形成金叉做为买入信号"""
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

        if len(fast_line) >= 2 and len(slow_line) >= 2:
            # 今天的快线值
            fast_current = fast_line[-1]
            # 昨天的快线值
            fast_last = fast_line[-2]
            # 今天的慢线值
            slow_current = slow_line[-1]
            # 昨天的慢线值
            slow_last = slow_line[-2]

            if slow_last < fast_last and fast_current < slow_current:
                # 快线下穿慢线, 形成卖出金叉，使用了今天收盘价格，明天买入
                return self.buy_next()
