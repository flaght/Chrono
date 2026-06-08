# -*- encoding:utf-8 -*-
"""
kx022 - 分析师预期因子 (调用端) - 近似实现

研报来源: 真实超预期系列研究之三：从低预期里寻找超预期.pdf
实现状态: generated_approximate
近似说明: 基于历史收益模式的超预期识别
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx022 import kx022 as calc_kx022


class ImpulseKx022(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx022_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx022"

    def calc_impulse(self, kl_pd):
        """计算分析师预期因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx022_keys:
            factor = calc_kx022(
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
