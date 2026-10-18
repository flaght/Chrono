import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi026 import oi026 as calc_oi026


class ImpulseOi026(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi026_keys = default_keys

    @property
    def name(self):
        return "oi026"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi026_keys:
            oi026 = calc_oi026(openint=kl_pd['openint'],
                               value=kl_pd['value'],
                               open=kl_pd['open'],
                               high=kl_pd['high'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi026 = self._format(oi026, name=name)
            impulse_dict[name] = oi026
        return impulse_dict
