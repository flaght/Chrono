# -*- encoding:utf-8 -*-
"""
加权上影线频率因子
来源: 西南证券 - 因子选股系列：加权影线频率与K线形态因子
"""
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i016.core.ki027 import ki027 as calc_ki027


class ImpulseKi027(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki027_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki027"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki027_keys:
            ki027 = calc_ki027(close=kl_pd['close'],
                               open=kl_pd['open'],
                               high=kl_pd['high'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ki027 = self._format(ki027, name=name)
            impulse_dict[name] = ki027
        return impulse_dict
