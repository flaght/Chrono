# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i004.core.db002 import db002 as calc_db002


class ImpulseDb002(ImpulseBase):
    
        def __init__(self, **kwargs):
            default_keys = default_keys1 if not kwargs else kwargs.get('keys')
            self.db002_keys = frozenset(default_keys)  # window, weriod, ewm
    
        @property
        def name(self):
            return "db002"
    
        def calc_impulse(self, kl_pd):
            impulse_dict = {}
            for dk in self.db002_keys:
                db002 = calc_db002(close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
                name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
                db002 = self._format(db002, name=name)
                impulse_dict[name] = db002
            return impulse_dict