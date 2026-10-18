# -*- encoding:utf-8 -*-
"""
rpv001_wrapper.py — RPV 因子 Wrapper 外壳
仅负责参数解包与调用 Core，不含任何业务逻辑
"""
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.rpv001 import rpv001 as calc_rpv001


class ImpulseRpv001(ImpulseBase):

    def __init__(self, **kwargs):
        # 若未提供 keys，则使用默认参数组合：(window, weriod, rv_window, min_samples, ewm)
        default_keys = [(1, 30, 10, 15, 0)] if not kwargs else kwargs.get('keys')
        self.rpv001_keys = default_keys

    @property
    def name(self):
        return "rpv001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rpv001_keys:
            # 只做参数解包，绝不在此编写因子逻辑
            rpv001 = calc_rpv001(close=kl_pd['close'],
                                 volume=kl_pd['volume'],
                                 window=dk[0],
                                 weriod=dk[1],
                                 rv_window=dk[2],
                                 min_samples=dk[3],
                                 ewm=True if dk[4] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1], dk[2], dk[3], dk[4])
            rpv001 = self._format(rpv001, name=name)
            impulse_dict[name] = rpv001
        return impulse_dict