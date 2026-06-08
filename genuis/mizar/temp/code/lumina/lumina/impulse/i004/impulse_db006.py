# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i004.core.db006 import db006 as calc_db006


class ImpulseDb006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.db006_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "db006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.db006_keys:
            db006 = calc_db006(vwap=kl_pd['vwap'],
                               volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            db006 = self._format(db006, name=name)
            impulse_dict[name] = db006
        return impulse_dict
