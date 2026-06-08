# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i003.core.xy001 import xy001 as calc_xy001


class ImpulseXy001(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.xy001_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "xy001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.xy001_keys:
            xy001 = calc_xy001(volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            xy001 = self._format(xy001, name=name)
            impulse_dict[name] = xy001
        return impulse_dict