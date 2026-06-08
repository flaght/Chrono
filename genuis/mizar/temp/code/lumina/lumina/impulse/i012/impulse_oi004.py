import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi004 import oi004 as calc_oi004


class ImpulseOi004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi004_keys = default_keys

    @property
    def name(self):
        return "oi004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi004_keys:
            oi004 = calc_oi004(close=kl_pd['close'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi004 = self._format(oi004, name=name)
            impulse_dict[name] = oi004
        return impulse_dict
