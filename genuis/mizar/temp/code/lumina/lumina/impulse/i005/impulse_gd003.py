import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i005.core.gd003 import gd003 as calc_gd003


class ImpulseGd003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.gd003_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "gd003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.gd003_keys:
            gd003 = calc_gd003(value=kl_pd['value'],
                               volume=kl_pd['volume'],
                               open=kl_pd['open'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            gd003 = self._format(gd003, name=name)
            impulse_dict[name] = gd003
        return impulse_dict
