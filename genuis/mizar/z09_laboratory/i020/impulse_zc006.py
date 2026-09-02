# -*- encoding: utf-8 -*-
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from laboratory.i020.core.zc006 import zc006 as calc_zc006


class ImpulseZc006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.zc006_keys = default_keys

    @property
    def name(self):
        return "zc006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.zc006_keys:
            # dk = (window, weriod, ewm)
            zc006 = calc_zc006(close=kl_pd['close'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            zc006 = self._format(zc006, name=name)
            impulse_dict[name] = zc006
        return impulse_dict