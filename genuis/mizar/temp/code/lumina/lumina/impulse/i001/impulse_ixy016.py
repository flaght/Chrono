# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i001.core.ixy016 import ixy016 as calc_ixy016


class ImpulseIxy016(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy016_keys = default_keys

    @property
    def name(self):
        return "ixy016"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy016_keys:
            ixy016 = calc_ixy016(close=kl_pd['close'],
                             volume=kl_pd['volume'] / 1e6,
                             window=dk[0],
                             weriod=dk[1],
                             ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy016 = self._format(ixy016, name=name)
            impulse_dict[name] = ixy016
        return impulse_dict
