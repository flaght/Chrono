# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from laboratory.i020.core.zc00401 import zc00401 as calc_zc00401


class ImpulseZc00401(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.zc00401_keys = default_keys

    @property
    def name(self):
        return "zc00401"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc00401_keys:
            # dk = (window, fast, slow, weriod, ewm) 来自 default_keys3
            zc00401 = calc_zc00401(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                openint=kl_pd['openint'],
                window=dk[0],
                fast=dk[1],
                slow=dk[2],
                weriod=dk[3],
                ewm=True if dk[4] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(
                self.name, dk[0], dk[1], dk[2], dk[3], dk[4]
            )
            zc00401 = self._format(zc00401, name=name)
            impulse_dict[name] = zc00401
        return impulse_dict