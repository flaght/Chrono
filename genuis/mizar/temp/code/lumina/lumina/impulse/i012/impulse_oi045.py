import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi045 import oi045 as calc_oi045


class ImpulseOi045(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi045_keys = default_keys

    @property
    def name(self):
        return "oi045"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi045_keys:
            oi045 = calc_oi045(close=kl_pd['close'],
                                          open=kl_pd['open'],
                                          high=kl_pd['high'],
                                          low=kl_pd['low'],
                                          openint=kl_pd['openint'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi045 = self._format(oi045, name=name1)
            impulse_dict[name1] = oi045
        return impulse_dict
