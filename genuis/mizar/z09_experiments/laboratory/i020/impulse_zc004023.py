# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.zc004023 import zc004023 as calc_zc004023


class ImpulseZc004023(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.zc004023_keys = default_keys

    @property
    def name(self):
        return "zc004023"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc004023_keys:
            # dk = (window, weriod, ewm)
            zc004023 = calc_zc004023(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                openint=kl_pd['openint'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            zc004023 = self._format(zc004023, name=name)
            impulse_dict[name] = zc004023
        return impulse_dict