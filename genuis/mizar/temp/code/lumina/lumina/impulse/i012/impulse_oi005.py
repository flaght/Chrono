import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi005 import oi005 as calc_oi005


class ImpulseOi005(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi005_keys = default_keys

    @property
    def name(self):
        return "oi005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi005_keys:
            oi005 = calc_oi005(close=kl_pd['close'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi005 = self._format(oi005, name=name)
            impulse_dict[name] = oi005
        return impulse_dict
