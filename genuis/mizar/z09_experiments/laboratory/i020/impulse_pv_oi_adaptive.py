# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase, default_keys1
from laboratory.i020.core.pv_oi_adaptive import pv_oi_adaptive as calc_pv_oi_adaptive


class ImpulsePvOiAdaptive(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get("keys")
        self.pv_oi_adaptive_keys = default_keys

    @property
    def name(self):
        return "pv_oi_adaptive"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.pv_oi_adaptive_keys:
            res = calc_pv_oi_adaptive(
                close=kl_pd["close"],
                openint=kl_pd["openint"],
                volume=kl_pd["volume"],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False,
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            impulse_dict[name] = self._format(res, name=name)
        return impulse_dict
