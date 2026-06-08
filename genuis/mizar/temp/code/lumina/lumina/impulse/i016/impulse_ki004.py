# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki004 import ki004 as calc_ki004


class ImpulseKi004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki004_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki004_keys:
            factor = calc_ki004(
                open=kl_pd['open'],
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
