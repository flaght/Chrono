# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i002.core.cj002 import cj002 as calc_cj002


class ImpulseCj002(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cj002_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "cj002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cj002_keys:
            cj002 = calc_cj002(volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cj002 = self._format(cj002, name=name)
            impulse_dict[name] = cj002
        return impulse_dict
