import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi031 import oi031 as calc_oi031


class ImpulseOi031(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi031_keys = default_keys

    @property
    def name(self):
        return "oi031"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi031_keys:
            oi031 = calc_oi031(openint=kl_pd['openint'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi031 = self._format(oi031, name=name)
            impulse_dict[name] = oi031
        return impulse_dict
