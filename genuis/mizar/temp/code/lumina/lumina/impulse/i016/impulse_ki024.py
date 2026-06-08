# -*- encoding:utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from .core.ki024 import ki024 as calc_ki024


class ImpulseKi024(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.ki024_keys = frozenset(default_keys)

    @property
    def name(self):
        return "ki024"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ki024_keys:
            ki024 = calc_ki024(
                close=kl_pd['close'],
                volume=kl_pd['volume'],
                openint=kl_pd['openint'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False
            )
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            ki024 = self._format(ki024, name=name)
            impulse_dict[name] = ki024
        return impulse_dict
