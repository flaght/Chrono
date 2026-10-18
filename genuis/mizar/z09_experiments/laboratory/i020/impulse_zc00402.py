# -*- encoding:utf-8 -*-
"""
wrapper/impulse_zc00402.py
JUVP-DIVERGENCE 量仓背离状态条件增量因子 Wrapper 外壳
"""
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from laboratory.i020.core.zc00402 import zc00402 as calc_zc00402


class ImpulseZc00402(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.zc00402_keys = default_keys

    @property
    def name(self):
        return "zc00402"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc00402_keys:
            # dk = (window, fast, slow, weriod, ewm) 来自 default_keys3
            zc00402 = calc_zc00402(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                openint=kl_pd['openint'],
                window=dk[0],
                fast=dk[1],
                slow=dk[2],
                weriod=dk[3],
                ewm=True if dk[4] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(
                self.name, dk[0], dk[1], dk[2], dk[3], dk[4])
            zc00402 = self._format(zc00402, name=name)
            impulse_dict[name] = zc00402
        return impulse_dict