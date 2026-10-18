# -*- encoding:utf-8 -*-
"""
kx049 - 高频量价因子 (调用端)

研报来源: 量化研究系列报告之十九：破解Alpha投资困境，因子择时方案再探索.pdf
实现状态: generated
数据字段: close, volume
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx049 import kx049 as calc_kx049


class ImpulseKx049(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx049_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx049"

    def calc_impulse(self, kl_pd):
        """计算高频量价因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx049_keys:
            factor = calc_kx049(
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
