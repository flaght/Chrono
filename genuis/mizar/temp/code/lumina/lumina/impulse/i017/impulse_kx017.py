# -*- encoding:utf-8 -*-
"""
kx017 - 因子加权过程中的大类权重控制 (调用端) - 近似实现

研报来源: 因子选股系列报告之六十八：因子加权过程中的大类权重控制.pdf
实现状态: generated_approximate
近似说明: 基于日频单股票特征近似多因子权重控制策略
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx017 import kx017 as calc_kx017


class ImpulseKx017(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx017_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx017"

    def calc_impulse(self, kl_pd):
        """计算因子权重控制因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx017_keys:
            factor = calc_kx017(
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
