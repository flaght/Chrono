# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i002.core.cj010 import cj010 as calc_cj010


class ImpulseCj010(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj010_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "cj010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj010_keys:
            cj010 = calc_cj010(open=kl_pd['open'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj010 = self._format(cj010, name=name)
            impulse_dict[name] = cj010
        return impulse_dict
