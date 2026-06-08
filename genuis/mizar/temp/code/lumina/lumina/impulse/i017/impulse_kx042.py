# -*- encoding:utf-8 -*-
"""
kx042 - TMT行业量化选股因子 (量化评论系列)
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx042 import kx042 as calc_kx042


class ImpulseKx042(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx042_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx042"

    def calc_impulse(self, kl_pd):
        """计算TMT选股因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx042_keys:
            factor = calc_kx042(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
