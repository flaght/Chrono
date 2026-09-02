# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.volatility002 import volatility002 as calc_volatility002


class ImpulseVolatility002(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = [(3, 10, 1), (5, 15, 1), (5, 15, 0)]#default_keys1 if not kwargs else kwargs.get('keys')
        self.volatility002_keys = default_keys

    @property
    def name(self):
        return "volatility002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.volatility002_keys:
            # 只做参数解包，绝对没有因子业务逻辑
            # dk 预期为 (window, weriod, lookback) 或 (window, weriod, lookback, lambda_param, ...)
            # 以下按 (window, weriod, lookback) 解包，其他参数使用默认值
            volatility002 = calc_volatility002(close=kl_pd['close'],
                                                window=dk[0],
                                                weriod=dk[1],
                                                lookback=5,
                                                ewm=dk[2])
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            volatility002 = self._format(volatility002, name=name)
            impulse_dict[name] = volatility002
        return impulse_dict