# -*- encoding:utf-8 -*-
"""
kx015 - 基于大单的alpha因子构建 (调用端) - 近似实现

研报来源: 因子选股系列之七十九：基于大单的alpha因子构建.pdf
实现状态: generated_approximate
近似说明: 基于日频成交量激增事件和价格冲击近似大单交易行为
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx015 import kx015 as calc_kx015


class ImpulseKx015(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx015_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx015"

    def calc_impulse(self, kl_pd):
        """计算基于大单的alpha因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx015_keys:
            factor = calc_kx015(
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
