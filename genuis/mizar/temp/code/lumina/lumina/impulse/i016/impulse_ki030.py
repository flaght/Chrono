# -*- encoding:utf-8 -*-
"""
信息分布涨跌幅因子
来源: 西南证券 - 求索动量因子系列：反应不足or反应过度？从信息分布到动量反转
"""
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i016.core.ki030 import ki030 as calc_ki030


class ImpulseKi030(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki030_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki030"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki030_keys:
            ki030 = calc_ki030(close=kl_pd['close'],
                               volume=kl_pd['volume'] / 1e6,
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ki030 = self._format(ki030, name=name)
            impulse_dict[name] = ki030
        return impulse_dict
