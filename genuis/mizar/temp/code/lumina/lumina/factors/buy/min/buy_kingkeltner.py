# -*- encoding:utf-8 -*-
import pdb, math
import numpy as np
from ultron.ump.technical.line import Line
from ultron.ump.indicator.ma import calc_ma_from_prices
from ultron.ump.indicator.rsi import _calc_rsi_from_pd
from ultron.ump.indicator.atr import calc_atr
from ultron.ump.core.helper import pd_rolling_min, pd_rolling_max, pd_resample
from ultron.ump.indicator.ma import EMACalcType
from lumina.factors.buy.fixes import FactorBuyID, BuyCallMixin, BuyPutMixin


class FactorKeltnerBuy(FactorBuyID):

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：xd: 均线周期，默认不设置，使用自适应动态快线
        """
        self.ma_xd = kwargs.pop('ma_xd', -1)
        self.atr_xd = kwargs.pop('atr_xd', -1)
        self.dynamic_ma = False
        self.dynamic_atr = False
        if self.ma_xd == -1:
            self.ma_xd = 40
            self.dynamic_ma = True

        if self.atr_xd == -1:
            self.atr_xd = 20
            self.dynamic_atr = True

        # 动态可设置参数重采样周期最大值，默认90
        self.resample_max = kwargs.pop('resample_max', 10)
        # 动态可设置参数重采样周期最小值，默认10
        self.resample_min = kwargs.pop('resample_min', 2)
        # 动态可设置参数代表慢线的选取阀值，默认0.12
        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        self.ewm = kwargs.pop('ewm', 1)

        kwargs['xd'] = self.ma_xd + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(FactorKeltnerBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:ma={},atr={},change_threshold={}'.format(
            self.__class__.__name__, self.ma_xd, self.atr_xd,
            self.change_threshold)

    def _dynamic_calc_xd(self, bar):
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

    ## 通过天数据刷新min 拟合参数
    def fit_day(self, bar):
        self.ma_xd = self._dynamic_calc_xd(bar)
        self.atr_xd = self._dynamic_calc_xd(bar)
        self.factor_name = '{}:ma={},atr={},xd={}'.format(
            self.__class__.__name__, self.ma_xd, self.atr_xd, self.xd)


class FactorKingKeltnerBuyL(FactorKeltnerBuy, BuyCallMixin):

    def fit_bar(self, bar):
        # 计算均线
        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA
        ma_line = calc_ma_from_prices(
            (self.xd_kl.close + self.xd_kl.high + self.xd_kl.low) / 3,
            int(self.ma_xd),
            min_periods=1,
            from_calc=from_calc)

        atr_line = calc_atr(self.xd_kl.high, self.xd_kl.low, self.xd_kl.close,
                            int(self.atr_xd))

        # 三价均线向上，并且价格上破通道上轨，开多单
        if ma_line[-1] > ma_line[-2] and bar.high > ma_line[-1] + atr_line[-1]:
            return self.buy_next()


class FactorKingKeltnerBuyS(FactorKeltnerBuy, BuyPutMixin):

    def fit_bar(self, bar):
        # 计算均线
        from_calc = EMACalcType.E_MA_EMA if self.ewm == 1 else EMACalcType.E_MA_MA
        ma_line = calc_ma_from_prices(
            (self.xd_kl.close + self.xd_kl.high + self.xd_kl.low) / 3,
            int(self.ma_xd),
            min_periods=1,
            from_calc=from_calc)

        atr_line = calc_atr(self.xd_kl.high, self.xd_kl.low, self.xd_kl.close,
                            int(self.atr_xd))

        # 三价均线向下，并且价格下破通道下轨，开空单
        if ma_line[-1] < ma_line[-2] and bar.low > ma_line[-1] - atr_line[-1]:
            return self.buy_next()
