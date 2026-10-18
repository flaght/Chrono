# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from .core.ki018 import ki018 as calc_ki018


class ImpulseKi018(ImpulseBase):
    """
    多空对比因子 (Bull-Bear Ratio)
    来源: 华西证券《基于量价因子的ETF组合策略》
    """

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.ki018_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki018"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki018_keys:
            factor = calc_ki018(
                close=kl_pd['close'],
                high=kl_pd['high'],
                low=kl_pd['low'],
                volume=kl_pd['volume'],
                window=dk[0],
                fast=dk[1],
                slow=dk[2],
                ewm=True if dk[3] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
