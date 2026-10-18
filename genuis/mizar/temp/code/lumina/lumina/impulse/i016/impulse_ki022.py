# -*- encoding:utf-8 -*-
"""
二阶动量因子
来源: 华西证券 - 基于量价因子的ETF组合策略
"""
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from lumina.impulse.i016.core.ki022 import ki022 as calc_ki022


class ImpulseKi022(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.ki022_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki022"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki022_keys:
            ki022 = calc_ki022(close=kl_pd['close'],
                               window=dk[0],
                               fast=dk[1],
                               slow=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3])
            ki022 = self._format(ki022, name=name)
            impulse_dict[name] = ki022
        return impulse_dict
