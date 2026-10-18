# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i004.core.db005 import db005 as calc_db005


class ImpulseDb005(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.db005_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "db005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.db005_keys:
            db005 = calc_db005(close=kl_pd['close'],
                               quant=0.2,
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            db005 = self._format(db005, name=name)
            impulse_dict[name] = db005
        return impulse_dict
