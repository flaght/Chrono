import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi001 import oi001 as calc_oi001


class ImpulseOi001(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi001_keys = default_keys

    @property
    def name(self):
        return "oi001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi001_keys:
            oi001 = calc_oi001(close=kl_pd['close'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi001 = self._format(oi001, name=name)
            impulse_dict[name] = oi001
        return impulse_dict
