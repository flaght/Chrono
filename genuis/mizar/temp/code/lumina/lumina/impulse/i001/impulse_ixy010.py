# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i001.core.ixy010 import ixy010 as calc_ixy010


class ImpulseIxy010(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy010_keys = default_keys

    @property
    def name(self):
        return "ixy010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy010_keys:
            ixy010 = calc_ixy010(volume=kl_pd['volume'] / 1e6,
                                   window=dk[0],
                                   weriod=dk[1],
                                   ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy010 = self._format(ixy010, name=name)
            impulse_dict[name] = ixy010
        return impulse_dict
