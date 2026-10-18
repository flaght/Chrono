# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i001.core.ixy006 import ixy006 as calc_ixy006


class ImpulseIxy006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy006_keys = default_keys

    @property
    def name(self):
        return "ixy006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy006_keys:
            ixy006 = calc_ixy006(close=kl_pd['close'],
                                 volume=kl_pd['volume'] / 1e6,
                                 window=dk[0],
                                 weriod=dk[1],
                                 ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy006 = self._format(ixy006, name=name)
            impulse_dict[name] = ixy006
        return impulse_dict
