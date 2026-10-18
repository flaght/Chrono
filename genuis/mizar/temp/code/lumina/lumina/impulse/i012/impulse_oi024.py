import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi024 import oi024 as calc_oi024

class ImpulseOi024(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi024_keys = default_keys

    @property
    def name(self):
        return "oi024"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi024_keys:
            oi024 = calc_oi024(openint=kl_pd['openint'],
                               close=kl_pd['close'],
                               value=kl_pd['value'],
                               open=kl_pd['open'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi024 = self._format(oi024, name=name)
            impulse_dict[name] = oi024
        return impulse_dict