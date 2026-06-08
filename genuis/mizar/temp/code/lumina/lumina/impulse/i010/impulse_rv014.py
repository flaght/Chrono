import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv014 import rv014 as calc_rv014


class ImpulseRv014(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = [(0.75, 5, 10, 1), (0.75, 10, 15, 1), (0.75, 5, 10, 0),
                        (0.75, 10, 15, 0), (0.25, 5, 10, 1), (0.25, 10, 15, 1),
                        (0.25, 5, 10, 0),
                        (0.25, 10, 15,
                         0)] if not kwargs else kwargs.get('keys')
        self.rv011_keys = default_keys

    @property
    def name(self):
        return "rv014"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv011_keys:
            rv014_1, rv014_2 = calc_rv014(volume=kl_pd['volume'],
                                          threshold=dk[0],
                                          window=dk[1],
                                          weriod=dk[2],
                                          ewm=True if dk[3] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, int(dk[0] * 100),
                                                 dk[1], dk[2], dk[3], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, int(dk[0] * 100),
                                                 dk[1], dk[2], dk[3], 2)

            rv014_1 = self._format(rv014_1, name=name1)
            rv014_2 = self._format(rv014_2, name=name2)
            impulse_dict[name1] = rv014_1
            impulse_dict[name2] = rv014_2
        return impulse_dict
