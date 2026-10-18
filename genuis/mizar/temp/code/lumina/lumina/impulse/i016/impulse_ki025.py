# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki025 import ki025 as calc_ki025


class ImpulseKi025(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki025_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki025"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki025_keys:
            ki025 = calc_ki025(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                openint=kl_pd['openint'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ki025 = self._format(ki025, name=name)
            impulse_dict[name] = ki025
        return impulse_dict
