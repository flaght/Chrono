import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi007 import oi007 as calc_oi007


class ImpulseOi007(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi007_keys = default_keys

    @property
    def name(self):
        return "oi007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi007_keys:
            oi007 = calc_oi007(openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi007 = self._format(oi007, name=name)
            impulse_dict[name] = oi007
        return impulse_dict
