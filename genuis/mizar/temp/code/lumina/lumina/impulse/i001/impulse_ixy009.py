# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i001.core.ixy009 import ixy009 as calc_ixy009


class ImpulseIxy009(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy009_keys = default_keys

    @property
    def name(self):
        return "ixy009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy009_keys:
            ixy009 = calc_ixy009(close=kl_pd['close'],
                             volume=kl_pd['volume'],
                             window=dk[0],
                             weriod=dk[1],
                             ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy009 = self._format(ixy009, name=name)
            impulse_dict[name] = ixy009
        return impulse_dict
