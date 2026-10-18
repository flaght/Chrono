# -*- encoding:utf-8 -*-
"""
VPIN指令流毒性因子
来源: 招商证券 - "琢璞"系列报告之十七：高频数据中的知情交易（二）
"""
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki032 import ki032 as calc_ki032


class ImpulseKi032(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki032_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki032"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki032_keys:
            factor = calc_ki032(
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
