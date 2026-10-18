# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from .core.kx030 import kx030 as calc_kx030

class ImpulseKx030(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(window, fast, slow, weriod, ewm)
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.kx030_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx030"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.kx030_keys:
            factor = calc_kx030(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                fast=dk[1],
                slow=dk[2],
                weriod=dk[3],
                window=dk[0],
                ewm=True if dk[4] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1], dk[2], dk[3], dk[4])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
