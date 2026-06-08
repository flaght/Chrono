# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i004.core.db003 import db003 as calc_db003

class ImpulseDb003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.db003_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "db003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.db003_keys:
            db003 = calc_db003(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            db003 = self._format(db003, name=name)
            impulse_dict[name] = db003
        return impulse_dict