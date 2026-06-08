import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi023 import oi023 as calc_oi023


class ImpulseOi023(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi023_keys = default_keys

    @property
    def name(self):
        return "oi023"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi023_keys:
            oi023 = calc_oi023(openint=kl_pd['openint'],
                               close=kl_pd['close'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               open=kl_pd['open'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi023 = self._format(oi023, name=name)
            impulse_dict[name] = oi023
        return impulse_dict
