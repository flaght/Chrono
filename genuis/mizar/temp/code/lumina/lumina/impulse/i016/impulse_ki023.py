# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki023 import ki023 as calc_ki023


class ImpulseKi023(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki023_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki023"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki023_keys:
            ki023 = calc_ki023(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                openint=kl_pd['openint'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ki023 = self._format(ki023, name=name)
            impulse_dict[name] = ki023
        return impulse_dict
