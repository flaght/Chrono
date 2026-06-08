import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi028 import oi028 as calc_oi028


class ImpulseOi028(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi028_keys = default_keys

    @property
    def name(self):
        return "oi028"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi028_keys:
            oi028 = calc_oi028(high=kl_pd['high'],
                               open=kl_pd['open'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi028 = self._format(oi028, name=name)
            impulse_dict[name] = oi028
        return impulse_dict
