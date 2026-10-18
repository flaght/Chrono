# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from laboratory.i020.core.tmp001 import tmp001 as calc_tmp001


class ImpulseTmp001(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.tmp001_keys = default_keys

    @property
    def name(self):
        return "tmp001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.tmp001_keys:
            # dk = (window, fast, slow, ewm) 来自 default_keys2
            tmp001 = calc_tmp001(close=kl_pd['close'],
                                 openint=kl_pd['openint'],
                                 window=dk[0],
                                 fast=dk[1],
                                 slow=dk[2],
                                 ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2],
                                                dk[3])
            tmp001 = self._format(tmp001, name=name)
            impulse_dict[name] = tmp001
        return impulse_dict