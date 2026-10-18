import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv013 import rv013 as calc_rv013


class ImpulseRv013(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = [(0.75, 5, 10, 1), (0.75, 10, 15, 1), (0.75, 5, 10, 0),
                        (0.75, 10, 15, 0), (0.25, 5, 10, 1), (0.25, 10, 15, 1),
                        (0.25, 5, 10, 0),
                        (0.25, 10, 15,
                         0)] if not kwargs else kwargs.get('keys')
        self.rv011_keys = default_keys

    @property
    def name(self):
        return "rv013"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv011_keys:
            rv013_1, rv013_2 = calc_rv013(close=kl_pd['close'],
                                          volume=kl_pd['volume'],
                                          threshold=dk[0],
                                          window=dk[1],
                                          weriod=dk[2],
                                          ewm=True if dk[3] == 1 else False)
            ## FIXME: 这里的name有问题 少一位
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, int(dk[0] * 100),
                                                 dk[1], dk[2], dk[3], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, int(dk[0] * 100),
                                                 dk[1], dk[2], dk[3], 2)

            rv013_1 = self._format(rv013_1, name=name1)
            rv013_2 = self._format(rv013_2, name=name2)
            impulse_dict[name1] = rv013_1
            impulse_dict[name2] = rv013_2
        return impulse_dict

    @classmethod
    def serializ(cls, params):
        params = [(dk[0] / 100.0, dk[1], dk[2], dk[3]) for dk in params]
        return params
