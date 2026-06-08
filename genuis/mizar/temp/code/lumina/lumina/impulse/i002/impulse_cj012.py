# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i002.core.cj012 import cj012 as calc_cj012


class ImpulseCj012(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj012_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "cj012"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj012_keys:
            cj012 = calc_cj012(close=kl_pd['close'],
                               volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj012 = self._format(cj012, name=name)
            impulse_dict[name] = cj012
        return impulse_dict
