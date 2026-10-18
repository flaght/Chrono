# -*- encoding:utf-8 -*-
"""
kx040 - 高股息预期因子 (选股系列)
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx040 import kx040 as calc_kx040


class ImpulseKx040(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm) - 较长周期
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx040_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx040"

    def calc_impulse(self, kl_pd):
        """计算高股息预期因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx040_keys:
            factor = calc_kx040(
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
