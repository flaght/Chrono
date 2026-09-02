# -*- coding: utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.zc007 import zc007 as calc_zc007


class ImpulseZc007(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.zc007_keys = default_keys

    @property
    def name(self):
        return "zc007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc007_keys:
            zc007 = calc_zc007(
                close=kl_pd['close'],
                high=kl_pd['high'],
                low=kl_pd['low'],
                volume=kl_pd['volume'],
                openint=kl_pd['openint'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            zc007 = self._format(zc007, name=name)
            impulse_dict[name] = zc007
        return impulse_dict