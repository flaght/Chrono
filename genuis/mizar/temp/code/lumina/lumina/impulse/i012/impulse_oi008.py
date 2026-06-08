import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi008 import oi008 as calc_oi008


class ImpulseOi008(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi008_keys = default_keys

    @property
    def name(self):
        return "oi008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi008_keys:
            oi008 = calc_oi008(openint=kl_pd['openint'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi008 = self._format(oi008, name=name)
            impulse_dict[name] = oi008
        return impulse_dict
