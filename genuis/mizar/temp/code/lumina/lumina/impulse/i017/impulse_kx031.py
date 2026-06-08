# -*- encoding:utf-8 -*-
"""
kx031 - 估值错配因子 (寻找业绩与估值的错配)
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx031 import kx031 as calc_kx031


class ImpulseKx031(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx031_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx031"

    def calc_impulse(self, kl_pd):
        """计算估值错配因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx031_keys:
            factor = calc_kx031(
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
