import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv006 import rv006 as calc_rv006


class ImpulseRv006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv006_keys = default_keys

    @property
    def name(self):
        return "rv006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv006_keys:
            rv006_1, rv006_2 = calc_rv006(close=kl_pd['close'],
                                          open=kl_pd['open'],
                                          low=kl_pd['low'],
                                          high=kl_pd['high'],
                                          volume=kl_pd['volume'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 2)
            rv006_1 = self._format(rv006_1, name=name1)
            rv006_2 = self._format(rv006_2, name=name2)
            impulse_dict[name1] = rv006_1
            impulse_dict[name2] = rv006_2
        return impulse_dict
