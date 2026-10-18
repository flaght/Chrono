# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i001.core.ixy005 import ixy005 as calc_ixy005


class ImpulseIxy005(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy005_keys = default_keys

    @property
    def name(self):
        return "ixy005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy005_keys:
            ixy005 = calc_ixy005(close=kl_pd['close'],
                                 volume=kl_pd['volume'],
                                 window=dk[0],
                                 weriod=dk[1],
                                 ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy005 = self._format(ixy005, name=name)
            impulse_dict[name] = ixy005
        return impulse_dict
