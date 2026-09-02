# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from laboratory.i020.core.zc005 import zc005 as calc_zc005


class ImpulseZc005(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = [
                     (5, 5, 10, 10, 1), (5, 5, 10, 10, 0), (5, 5, 10, 15, 1),
                     (5, 5, 10, 15, 0), (10, 5, 10, 15, 1), (10, 5, 10, 15, 0)]
        self.zc005_keys = default_keys

    @property
    def name(self):
        return "zc005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc005_keys:
            zc005 = calc_zc005(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                amount=kl_pd['value'],
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
            zc005 = self._format(zc005, name=name)
            impulse_dict[name] = zc005
        return impulse_dict