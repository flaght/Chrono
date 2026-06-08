# -*- encoding:utf-8 -*-
"""
kx020 - 行业轮动因子 (调用端) - 近似实现

研报来源: 指数增强如何受益于行业轮动.pdf
实现状态: generated_approximate
近似说明: 基于股票表现的聚类分析近似行业轮动
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from .core.kx020 import kx020 as calc_kx020


class ImpulseKx020(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.kx020_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx020"

    def calc_impulse(self, kl_pd):
        """计算行业轮动因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx020_keys:
            factor = calc_kx020(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                window=dk[0],
                fast=dk[1],
                slow=dk[2],
                weriod=dk[3],
                ewm=True if dk[4] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1], dk[2], dk[3], dk[4])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
