# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i005.core.fz002 import fz002 as calc_fz002


class ImpulseFz002(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.fz002_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "fz002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.fz002_keys:
            fz002 = calc_fz002(close=kl_pd['close'],
                               volume=kl_pd['volume'] / 1e6,
                               quant=0.02,
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            fz002 = self._format(fz002, name=name)
            impulse_dict[name] = fz002
        return impulse_dict
