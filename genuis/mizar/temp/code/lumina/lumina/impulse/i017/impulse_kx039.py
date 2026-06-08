# -*- encoding:utf-8 -*-
"""
kx039 - 流通股本分布因子 (技术分析系列)
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx039 import kx039 as calc_kx039


class ImpulseKx039(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx039_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx039"

    def calc_impulse(self, kl_pd):
        """计算流通股本分布因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx039_keys:
            factor = calc_kx039(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
