import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv008 import rv008 as calc_rv008


class ImpulseRv008(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv008_keys = default_keys

    @property
    def name(self):
        return "rv008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv008_keys:
            rv008_1, rv008_2 = calc_rv008(high=kl_pd['high'],
                                          low=kl_pd['low'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 2)
            rv008_1 = self._format(rv008_1, name=name1)
            rv008_2 = self._format(rv008_2, name=name2)
            impulse_dict[name1] = rv008_1
            impulse_dict[name2] = rv008_2
        return impulse_dict
