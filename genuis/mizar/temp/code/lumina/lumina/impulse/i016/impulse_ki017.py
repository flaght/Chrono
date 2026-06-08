# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki017 import ki017 as calc_ki017


class ImpulseKi017(ImpulseBase):
    """
    量价背离因子 (Price-Volume Divergence)
    来源: 华西证券《基于量价因子的ETF组合策略》
    """

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki017_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki017"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki017_keys:
            factor = calc_ki017(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                openint=kl_pd['openint'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
