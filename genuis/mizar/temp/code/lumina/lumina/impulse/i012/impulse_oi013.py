import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi013 import oi013 as calc_oi013


class ImpulseOi013(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi013_keys = default_keys

    @property
    def name(self):
        return "oi013"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi013_keys:
            oi013 = calc_oi013(openint=kl_pd['openint'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi013 = self._format(oi013, name=name)
            impulse_dict[name] = oi013
        return impulse_dict
