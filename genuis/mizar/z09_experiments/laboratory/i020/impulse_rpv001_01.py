"""
rpv001_01_wrapper.py — Wrapper外壳，仅做参数解包与调用
"""
# -*- encoding:utf-8 -*-
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.rpv001_01 import rpv001_01 as calc_rpv001_01

class ImpulseRpv001_01(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制不使用 super().__init__(**kwargs)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rpv001_01_keys = default_keys

    @property
    def name(self):
        return "rpv001_01"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rpv001_01_keys:
            # 只做参数解包，无业务逻辑
            rpv001_01 = calc_rpv001_01(close=kl_pd['close'],
                                        high=kl_pd['high'],
                                        low=kl_pd['low'],
                                        window=dk[0],
                                        weriod=dk[1],
                                        ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            rpv001_01 = self._format(rpv001_01, name=name)
            impulse_dict[name] = rpv001_01
        return impulse_dict