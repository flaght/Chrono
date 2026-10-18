# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i002.core.cj014 import cj014 as calc_cj014


class ImpulseCj014(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj014_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "cj014"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj014_keys:
            cj014 = calc_cj014(close=kl_pd['close'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               open=kl_pd['open'],
                               vwap=kl_pd['vwap'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj014 = self._format(cj014, name=name)
            impulse_dict[name] = cj014
        return impulse_dict
