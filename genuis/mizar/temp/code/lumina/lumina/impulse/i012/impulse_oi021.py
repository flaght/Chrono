import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi021 import oi021 as calc_oi021

class ImpulseOi021(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi021_keys = default_keys

    @property
    def name(self):
        return "oi021"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi021_keys:
            oi021 = calc_oi021(openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi021 = self._format(oi021, name=name)
            impulse_dict[name] = oi021
        return impulse_dict