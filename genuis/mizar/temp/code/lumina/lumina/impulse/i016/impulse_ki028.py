# -*- encoding:utf-8 -*-
"""
加权下影线频率因子
来源: 西南证券 - 因子选股系列：加权影线频率与K线形态因子
"""
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i016.core.ki028 import ki028 as calc_ki028


class ImpulseKi028(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki028_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki028"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki028_keys:
            ki028 = calc_ki028(close=kl_pd['close'],
                               open=kl_pd['open'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ki028 = self._format(ki028, name=name)
            impulse_dict[name] = ki028
        return impulse_dict
