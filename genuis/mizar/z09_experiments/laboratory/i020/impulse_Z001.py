# -*- encoding:utf-8 -*-
"""
z001_wrapper.py — 量能时钟均匀度因子 Wrapper 外壳
仅做参数解包与调用 Core，不包含任何因子业务逻辑。
"""
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.Z001 import z001 as calc_z001


class ImpulseZ001(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.z001_keys = default_keys

    @property
    def name(self):
        return "z001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.z001_keys:
            # dk = (window, weriod, ewm) 来自 default_keys1
            # 辅助参数（bars_per_day, volume_buckets 等）使用 Core 默认值，不在此传递
            z001 = calc_z001(close=kl_pd['close'],
                              openint=kl_pd['openint'],
                              window=dk[0],
                              weriod=dk[1],
                              ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            z001 = self._format(z001, name=name)
            impulse_dict[name] = z001
        return impulse_dict