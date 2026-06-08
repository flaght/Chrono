# -*- encoding:utf-8 -*-
"""
kx045 - 行业泡沫拥挤度因子 (量化评论系列)
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx045 import kx045 as calc_kx045


class ImpulseKx045(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx045_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx045"

    def calc_impulse(self, kl_pd):
        """计算泡沫拥挤度因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx045_keys:
            factor = calc_kx045(
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
