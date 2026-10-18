# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i002.core.cj003 import cj003 as calc_cj003

class ImpulseCj003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj003_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "cj003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj003_keys:
            cj003 = calc_cj003(close=kl_pd['close'], volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj003 = self._format(cj003, name=name)
            impulse_dict[name] = cj003
        return impulse_dict