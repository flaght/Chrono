import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi044 import oi044 as calc_oi044


class ImpulseOi044(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi044_keys = default_keys

    @property
    def name(self):
        return "oi044"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi044_keys:
            oi044_1, oi044_2 = calc_oi044(close=kl_pd['close'],
                                          openint=kl_pd['openint'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2], 2)
            oi044_1 = self._format(oi044_1, name=name1)
            oi044_2 = self._format(oi044_2, name=name2)
            impulse_dict[name1] = oi044_1
            impulse_dict[name2] = oi044_2
        return impulse_dict
