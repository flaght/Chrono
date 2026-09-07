# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from laboratory.i020.core.zc00701 import zc00701 as calc_zc00701


class ImpulseZc00701(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.zc00701_keys = default_keys

    @property
    def name(self):
        return "zc00701"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc00701_keys:
            # dk = (window, fast, slow, weriod, ewm) 来自 default_keys3
            zc00701 = calc_zc00701(
                close=kl_pd['close'],
                high=kl_pd['high'],
                low=kl_pd['low'],
                volume=kl_pd['volume'],
                value=kl_pd['value'],
                openint=kl_pd['openint'],
                window=dk[0],
                fast=dk[1],
                slow=dk[2],
                weriod=dk[3],
                ewm=True if dk[4] == 1 else False,
            )
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(
                self.name, dk[0], dk[1], dk[2], dk[3], dk[4]
            )
            zc00701 = self._format(zc00701, name=name)
            impulse_dict[name] = zc00701
        return impulse_dict