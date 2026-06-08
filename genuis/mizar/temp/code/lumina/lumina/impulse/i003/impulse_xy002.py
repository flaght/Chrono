# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i003.core.xy002 import xy002 as calc_xy002

class ImpulseXy002(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.xy002_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "xy002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.xy002_keys:
            xy002 = calc_xy002(volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            xy002 = self._format(xy002, name=name)
            impulse_dict[name] = xy002
        return impulse_dict