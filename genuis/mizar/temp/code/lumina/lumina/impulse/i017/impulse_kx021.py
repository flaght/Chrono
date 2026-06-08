# -*- encoding:utf-8 -*-
"""
kx021 - 景气度轮动因子 (调用端) - 近似实现

研报来源: 指数研究与指数化投资系列：景气度视角下制造板块内部轮动配置策略.pdf
实现状态: generated_approximate
近似说明: 基于股票表现模式的景气度轮动
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.env import default_keys3
from .core.kx021 import kx021 as calc_kx021


class ImpulseKx021(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.kx021_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx021"

    def calc_impulse(self, kl_pd):
        """计算景气度轮动因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx021_keys:
            factor = calc_kx021(
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
