# -*- encoding:utf-8 -*-
"""
kx003 - 多层次订单失衡及订单斜率因子 (调用端)

研报来源: 因子深度研究系列：多层次订单失衡及订单斜率因子.pdf
实现状态: generated
数据字段: close, volume
"""

from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.kx003 import kx003 as calc_kx003


class ImpulseKx003(ImpulseBase):

    def __init__(self, **kwargs):
        # 自定义参数组合：(weriod, window, ewm)
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.kx003_keys = frozenset(default_keys)

    @property
    def name(self):
        return "kx003"

    def calc_impulse(self, kl_pd):
        """计算多层次订单失衡及订单斜率因子的所有参数组合"""
        impulse_dict = {}
        for dk in self.kx003_keys:
            factor = calc_kx003(
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
