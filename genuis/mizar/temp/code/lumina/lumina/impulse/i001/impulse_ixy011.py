# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i001.core.ixy011 import ixy011 as calc_ixy011


class ImpulseIxy011(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy011_keys = default_keys

    @property
    def name(self):
        return "ixy011"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy011_keys:
            ixy011 = calc_ixy011(volume=kl_pd['volume'] / 1e6,
                                   window=dk[0],
                                   weriod=dk[1],
                                   ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy011 = self._format(ixy011, name=name)
            impulse_dict[name] = ixy011
        return impulse_dict
