# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i004.core.db004 import db004 as calc_db004


class ImpulseDb004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.db004_keys = frozenset(default_keys)  # quant, window, weriod, ewm

    @property
    def name(self):
        return "db004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.db004_keys:
            db004 = calc_db004(close=kl_pd['close'],
                               quant=0.2,
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            db004 = self._format(db004, name=name)
            impulse_dict[name] = db004
        return impulse_dict
