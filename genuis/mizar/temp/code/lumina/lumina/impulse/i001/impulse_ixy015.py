# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i001.core.ixy015 import ixy015 as calc_ixy015


class ImpulseIxy015(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy015_keys = default_keys

    @property
    def name(self):
        return "ixy015"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy015_keys:
            ixy015 = calc_ixy015(close=kl_pd['close'],
                           window=dk[0],
                           weriod=dk[1],
                           ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy015 = self._format(ixy015, name=name)
            impulse_dict[name] = ixy015
        return impulse_dict
