# -*- encoding:utf-8 -*-
"""
kx027 - 资金流选股因子 (调用端) - 近似实现

研报来源: 资金流选股因子：主力资金杠杆效率.pdf
实现状态: generated_approximate
近似说明: 基于成交量和价格变动的资金流近似
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx027 import kx027 as calc_kx027


class ImpulseKx027(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx027_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx027"

    def calc_impulse(self, kl_pd):
        """计算资金流选股因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx027_keys:
            factor = calc_kx027(
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
