# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from laboratory.i020.core.zc003 import zc003 as calc_zc003


class ImpulseZc003(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.zc003_keys = default_keys

    @property
    def name(self):
        return "zc003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc003_keys:
            # dk = (window, fast, slow, weriod, ewm) 来自 default_keys3
            res = calc_zc003(close=kl_pd['close'],
                             volume=kl_pd['volume'],
                             window=dk[0],
                             fast=dk[1],
                             slow=dk[2],
                             weriod=dk[3],
                             ewm=True if dk[4] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1],
                                                    dk[2], dk[3], dk[4])
            res = self._format(res, name=name)
            impulse_dict[name] = res
        return impulse_dict