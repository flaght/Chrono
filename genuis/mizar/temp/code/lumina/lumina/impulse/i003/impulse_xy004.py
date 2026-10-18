# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i003.core.xy004 import xy004 as calc_xy004

class ImpulseXy004(ImpulseBase):
    
        def __init__(self, **kwargs):
            default_keys = default_keys1 if not kwargs else kwargs.get('keys')
            self.xy004_keys = frozenset(default_keys)  # window, weriod, ewm
    
        @property
        def name(self):
            return "xy004"
    
        def calc_impulse(self, kl_pd):
            impulse_dict = {}
            for dk in self.xy004_keys:
                xy004 = calc_xy004(volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
                name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
                xy004 = self._format(xy004, name=name)
                impulse_dict[name] = xy004
            return impulse_dict