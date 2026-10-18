import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from lumina.impulse.i012.core.oi032 import oi032 as calc_oi032


class ImpulseOi032(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.oi032_keys = default_keys

    @property
    def name(self):
        return "oi032"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi032_keys:
            oi032 = calc_oi032(high=kl_pd['high'],
                               low=kl_pd['low'],
                               close=kl_pd['close'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               fast=dk[1],
                               slow=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2],
                                                dk[3])
            oi032 = self._format(oi032, name=name)
            impulse_dict[name] = oi032
        return impulse_dict
