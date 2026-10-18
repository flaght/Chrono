# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki029 import ki029 as calc_ki029


class ImpulseKi029(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki029_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki029"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki029_keys:
            ki029 = calc_ki029(
                open=kl_pd['open'],
                high=kl_pd['high'],
                low=kl_pd['low'],
                close=kl_pd['close'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ki029 = self._format(ki029, name=name)
            impulse_dict[name] = ki029
        return impulse_dict
