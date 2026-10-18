# -*- encoding:utf-8 -*-
"""
kx002 - 分析师预期调整事件增强因子 (调用端)

研报来源: 因子深度研究系列：分析师预期调整事件增强选股策略全攻略.pdf
实现状态: generated
数据字段: close, volume
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx002 import kx002 as calc_kx002


class ImpulseKx002(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx002_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx002"

    def calc_impulse(self, kl_pd):
        """计算分析师预期调整事件增强因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx002_keys:
            factor = calc_kx002(
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
