import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi025 import oi025 as calc_oi025

class ImpulseOi025(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi025_keys = default_keys

    @property
    def name(self):
        return "oi025"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi025_keys:
            oi025 = calc_oi025(openint=kl_pd['openint'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi025 = self._format(oi025, name=name)
            impulse_dict[name] = oi025
        return impulse_dict