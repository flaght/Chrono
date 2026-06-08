import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi022 import oi022 as calc_oi022


class ImpulseOi022(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi022_keys = default_keys

    @property
    def name(self):
        return "oi022"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi022_keys:
            oi022 = calc_oi022(openint=kl_pd['openint'],
                               close=kl_pd['close'],
                               quant=0.8,
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi022 = self._format(oi022, name=name)
            impulse_dict[name] = oi022
        return impulse_dict
