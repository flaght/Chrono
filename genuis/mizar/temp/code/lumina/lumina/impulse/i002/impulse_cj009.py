# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i002.core.cj009 import cj009 as calc_cj009


class ImpulseCj009(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj009_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "cj009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj009_keys:
            cj009 = calc_cj009(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj009 = self._format(cj009, name=name)
            impulse_dict[name] = cj009
        return impulse_dict
