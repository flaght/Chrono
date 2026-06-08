# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki010 import ki010 as calc_ki010

class ImpulseKi010(ImpulseBase):
    """
    ATR突破因子

    基于平均真实波幅(ATR)标准化的日内动量因子，衡量价格变动相对于波动率的强度。
    ATR标准化可以使因子在不同波动率环境下保持可比性。
    """

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki010_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki010_keys:
            factor = calc_ki010(
                open=kl_pd['open'],
                high=kl_pd['high'],
                low=kl_pd['low'],
                close=kl_pd['close'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            factor = self._format(factor, name=name)
            impulse_dict[name] = factor
        return impulse_dict
