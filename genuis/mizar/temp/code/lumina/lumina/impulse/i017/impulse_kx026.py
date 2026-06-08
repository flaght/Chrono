# -*- encoding:utf-8 -*-
"""
kx026 - VIX情绪择时因子 (调用端) - 近似实现

研报来源: 指增中性专题报告（一）：基于情绪指标VIX的择时策略.pdf
实现状态: generated_approximate
近似说明: 基于价格波动率的VIX情绪近似
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx026 import kx026 as calc_kx026


class ImpulseKx026(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx026_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx026"

    def calc_impulse(self, kl_pd):
        """计算VIX情绪择时因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx026_keys:
            factor = calc_kx026(
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
