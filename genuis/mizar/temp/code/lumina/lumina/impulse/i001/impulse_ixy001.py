# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
#from lumina.impulse.i001.core.ixy001 import alpha125 as calc_alpha125
from lumina.impulse.i001.core.ixy001 import ixy001 as calc_ixy001


class ImpulseIxy001(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ixy001_keys = default_keys

    @property
    def name(self):
        return "ixy001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ixy001_keys:
            ixy001 = calc_ixy001(close=kl_pd['close'],
                                  window=dk[0],
                                  weriod=dk[1],
                                  ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ixy001 = self._format(ixy001, name=name)
            impulse_dict[name] = ixy001
        return impulse_dict
