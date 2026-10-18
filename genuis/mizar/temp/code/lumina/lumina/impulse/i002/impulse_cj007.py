# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i002.core.cj007 import cj007 as calc_cj007


class ImpulseCj007(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj007_keys = frozenset(default_keys)  # window, weriod, ewm


    @property
    def name(self):
        return "cj007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj007_keys:
            cj007 = calc_cj007(close=kl_pd['close'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj007 = self._format(cj007, name=name)
            impulse_dict[name] = cj007
        return impulse_dict
