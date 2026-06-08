# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i003.core.xy003 import xy003 as calc_xy003

class ImpulseXy003(ImpulseBase):
    
        def __init__(self, **kwargs):
            default_keys = default_keys1 if not kwargs else kwargs.get('keys')
            self.xy003_keys = frozenset(default_keys)  # window, weriod, ewm
    
        @property
        def name(self):
            return "xy003"
    
        def calc_impulse(self, kl_pd):
            impulse_dict = {}
            for dk in self.xy003_keys:
                xy003 = calc_xy003(volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
                name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
                xy003 = self._format(xy003, name=name)
                impulse_dict[name] = xy003
            return impulse_dict