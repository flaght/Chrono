import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi046 import oi046 as calc_oi046


class ImpulseOi046(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi046_keys = default_keys

    @property
    def name(self):
        return "oi046"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi046_keys:
            oi046 = calc_oi046(close=kl_pd['close'],
                                          open=kl_pd['open'],
                                          high=kl_pd['high'],
                                          low=kl_pd['low'],
                                          openint=kl_pd['openint'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi046 = self._format(oi046, name=name)
            impulse_dict[name] = oi046
        return impulse_dict
