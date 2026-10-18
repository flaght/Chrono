import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi016 import oi016 as calc_oi016


class ImpulseOi016(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi016_keys = default_keys

    @property
    def name(self):
        return "oi016"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi016_keys:
            oi016 = calc_oi016(close=kl_pd['close'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi016 = self._format(oi016, name=name)
            impulse_dict[name] = oi016
        return impulse_dict
