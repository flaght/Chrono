# -*- encoding:utf-8 -*-
"""
量幅同向因子
来源: 华西证券 - 基于量价因子的ETF组合策略
"""
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i016.core.ki021 import ki021 as calc_ki021


class ImpulseKi021(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki021_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki021"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki021_keys:
            ki021 = calc_ki021(high=kl_pd['high'],
                               low=kl_pd['low'],
                               volume=kl_pd['volume'] / 1e6,
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ki021 = self._format(ki021, name=name)
            impulse_dict[name] = ki021
        return impulse_dict
