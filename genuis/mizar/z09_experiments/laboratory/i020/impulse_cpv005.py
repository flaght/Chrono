# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.cpv005 import cpv005 as calc_cpv005


class ImpulseCpv005(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：不使用 super().__init__(**kwargs)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cpv005_keys = default_keys

    @property
    def name(self):
        return "cpv005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cpv005_keys:
            # 参数解包：dk[0]=window, dk[1]=weriod, dk[2]=h, dk[3]=ewm(0/1)
            cpv005 = calc_cpv005(close=kl_pd['close'],
                                 volume=kl_pd['volume'],
                                 openint=kl_pd['openint'],
                                 window=dk[0],
                                 weriod=dk[1],
                                 ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            cpv005 = self._format(cpv005, name=name)
            impulse_dict[name] = cpv005
        return impulse_dict