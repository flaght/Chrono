# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i001.core.ixy012 import ixy012 as calc_ixy012


class ImpulseIxy012(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy012_keys = default_keys

    @property
    def name(self):
        return "ixy012"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy012_keys:
            ixy012 = calc_ixy012(close=kl_pd['close'],
                                   window=dk[0],
                                   weriod=dk[1],
                                   ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy012 = self._format(ixy012, name=name)
            impulse_dict[name] = ixy012
        return impulse_dict
