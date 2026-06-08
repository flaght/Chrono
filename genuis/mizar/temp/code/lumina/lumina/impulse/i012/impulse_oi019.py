import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi019 import oi019 as calc_oi019


class ImpulseOi019(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi019_keys = default_keys

    @property
    def name(self):
        return "oi019"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi019_keys:
            oi019 = calc_oi019(openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi019 = self._format(oi019, name=name)
            impulse_dict[name] = oi019
        return impulse_dict
