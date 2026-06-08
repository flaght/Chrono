# -*- encoding:utf-8 -*-
"""
kx023 - Barra波动率因子 (调用端) - 近似实现

研报来源: Barra模型专题报告（一）：波动率因子.pdf
实现状态: generated_approximate
近似说明: 基于价格波动率的Barra风格因子实现
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx023 import kx023 as calc_kx023


class ImpulseKx023(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx023_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx023"

    def calc_impulse(self, kl_pd):
        """计算Barra波动率因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx023_keys:
            factor = calc_kx023(
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
