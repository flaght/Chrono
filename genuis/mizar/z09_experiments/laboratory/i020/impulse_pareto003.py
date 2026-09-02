# -*- encoding:utf-8 -*-
"""
pareto003_wrapper.py — Wrapper 外壳：参数解包 + 调用 Core
"""
import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from laboratory.i020.core.pareto003 import pareto003 as calc_pareto003


class ImpulsePareto003(ImpulseBase):

    def __init__(self, **kwargs):
        # 强制：绝不使用 super().__init__(**kwargs)
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.pareto003_keys = default_keys

    @property
    def name(self):
        return "pareto003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.pareto003_keys:
            # dk = (window, fast, slow, weriod, ewm) 来自 default_keys3
            pareto003 = calc_pareto003(
                openint=kl_pd['openint'],
                close=kl_pd['close'],
                window=dk[0],
                fast=dk[1],
                slow=dk[2],
                weriod=dk[3],
                ewm=True if dk[4] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(
                self.name, dk[0], dk[1], dk[2], dk[3], dk[4])
            pareto003 = self._format(pareto003, name=name)
            impulse_dict[name] = pareto003
        return impulse_dict