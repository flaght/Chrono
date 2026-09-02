# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from laboratory.i020.core.zc008 import zc008 as calc_zc008


class ImpulseZc008(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.zc008_keys = default_keys

    @property
    def name(self):
        return "zc008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc008_keys:
            # dk = (window, fast, slow, ewm) from default_keys2
            zc008 = calc_zc008(
                close=kl_pd['close'],
                high=kl_pd['high'],
                low=kl_pd['low'],
                volume=kl_pd['volume'],
                window=dk[0],
                fast=dk[1],
                slow=dk[2],
                ewm=True if dk[3] == 1 else False,
            )
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3])
            zc008 = self._format(zc008, name=name)
            impulse_dict[name] = zc008
        return impulse_dict