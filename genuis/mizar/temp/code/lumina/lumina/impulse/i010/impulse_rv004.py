import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv004 import rv004 as calc_rv004


class ImpulseRv004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv004_keys = default_keys

    @property
    def name(self):
        return "rv004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv004_keys:
            rv004_1, rv004_2 = calc_rv004(close=kl_pd['close'],
                                          open=kl_pd['open'],
                                          low=kl_pd['low'],
                                          high=kl_pd['high'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 2)
            rv004_1 = self._format(rv004_1, name=name1)
            rv004_2 = self._format(rv004_2, name=name2)
            impulse_dict[name1] = rv004_1
            impulse_dict[name2] = rv004_2
        return impulse_dict
