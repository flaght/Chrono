import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi040 import oi040 as calc_oi040


class ImpulseOi040(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi040_keys = default_keys

    @property
    def name(self):
        return "oi040"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi040_keys:
            oi040_1, oi040_2 = calc_oi040(close=kl_pd['close'],
                                          openint=kl_pd['openint'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2], 2)
            oi040_1 = self._format(oi040_1, name=name1)
            oi040_2 = self._format(oi040_2, name=name2)
            impulse_dict[name1] = oi040_1
            impulse_dict[name2] = oi040_2
        return impulse_dict
