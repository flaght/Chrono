# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.cpv006 import cpv006 as calc_cpv006


class ImpulseCpv006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.cpv006_keys = default_keys

    @property
    def name(self):
        return "cpv006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cpv006_keys:
            res = calc_cpv006(close=kl_pd['close'],
                              high=kl_pd['high'],
                              low=kl_pd['low'],
                              volume=kl_pd['volume'],
                              openint=kl_pd['openint'],
                              window=dk[0],
                              werid=dk[1],
                              ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            impulse_dict[name] = self._format(res, name=name)

        return impulse_dict
