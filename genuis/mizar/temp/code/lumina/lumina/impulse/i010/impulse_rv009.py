import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv009 import rv009 as calc_rv009


class ImpulseRv009(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv009_keys = default_keys

    @property
    def name(self):
        return "rv009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv009_keys:
            rv009_1, rv009_2 = calc_rv009(volume=kl_pd['volume'],
                                          close=kl_pd['close'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 2)
            rv009_1 = self._format(rv009_1, name=name1)
            rv009_2 = self._format(rv009_2, name=name2)
            impulse_dict[name1] = rv009_1
            impulse_dict[name2] = rv009_2
        return impulse_dict
