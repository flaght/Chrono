# -*- encoding:utf-8 -*-
"""
kx006 - 单商品指数编制概述及优化因子 (调用端)

研报来源: 因子与指数投资揭秘系列十二：单商品指数编制概述及优化.pdf
实现状态: generated
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx006 import kx006 as calc_kx006


class ImpulseKx006(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx006_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx006"

    def calc_impulse(self, kl_pd):
        """计算单商品指数编制优化因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx006_keys:
            factor = calc_kx006(
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
