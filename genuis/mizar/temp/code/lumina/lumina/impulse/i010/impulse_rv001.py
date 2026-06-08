import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv001 import rv001 as calc_rv001


class ImpulseRv001(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv001_keys = default_keys

    @property
    def name(self):
        return "rv001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv001_keys:
            rv001_1, rv001_2 = calc_rv001(close=kl_pd['close'],
                                          volume=kl_pd['volume'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 2)
            rv001_1 = self._format(rv001_1, name=name1)
            rv001_2 = self._format(rv001_2, name=name2)
            impulse_dict[name1] = rv001_1
            impulse_dict[name2] = rv001_2
        return impulse_dict
