import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv010 import rv010 as calc_rv010


class ImpulseRv010(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv010_keys = default_keys

    @property
    def name(self):
        return "rv010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv010_keys:
            rv010_1, rv010_2 = calc_rv010(high=kl_pd['high'],
                                          low=kl_pd['low'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 2)
            rv010_1 = self._format(rv010_1, name=name1)
            rv010_2 = self._format(rv010_2, name=name2)
            impulse_dict[name1] = rv010_1
            impulse_dict[name2] = rv010_2
        return impulse_dict
