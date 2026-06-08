# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i002.core.cj006 import cj006 as calc_cj006

class ImpulseCj006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj006_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "cj006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj006_keys:
            cj006 = calc_cj006(volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj006 = self._format(cj006, name=name)
            impulse_dict[name] = cj006
        return impulse_dict