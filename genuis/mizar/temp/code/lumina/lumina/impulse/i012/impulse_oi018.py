import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi018 import oi018 as calc_oi018


class ImpulseOi018(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi018_keys = default_keys

    @property
    def name(self):
        return "oi018"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi018_keys:
            oi018 = calc_oi018(openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi018 = self._format(oi018, name=name)
            impulse_dict[name] = oi018
        return impulse_dict
