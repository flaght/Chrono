# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.cj017 import cj017 as calc_cj017


class ImpulseCj017(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj017_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "cj017"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj017_keys:
            cj017 = calc_cj017(close=kl_pd['close'],
                               volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj017 = self._format(cj017, name=name)
            impulse_dict[name] = cj017
        return impulse_dict
