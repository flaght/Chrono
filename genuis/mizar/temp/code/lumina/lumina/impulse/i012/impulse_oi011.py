import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi011 import oi011 as calc_oi011


class ImpulseOi011(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi011_keys = default_keys

    @property
    def name(self):
        return "oi011"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi011_keys:
            oi011 = calc_oi011(close=kl_pd['close'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi011 = self._format(oi011, name=name)
            impulse_dict[name] = oi011
        return impulse_dict
