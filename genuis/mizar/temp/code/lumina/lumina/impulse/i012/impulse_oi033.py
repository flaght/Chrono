import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi033 import oi033 as calc_oi033


class ImpulseOi033(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi033_keys = default_keys

    @property
    def name(self):
        return "oi033"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi033_keys:
            oi033 = calc_oi033(openint=kl_pd['openint'],
                               low=kl_pd['low'],
                               close=kl_pd['close'],
                               high=kl_pd['high'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi033 = self._format(oi033, name=name)
            impulse_dict[name] = oi033
        return impulse_dict
