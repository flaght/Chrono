import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv003 import dv003 as calc_dv003


class ImpulseDv003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv003_keys = default_keys

    @property
    def name(self):
        return "dv003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv003_keys:
            dv003 = calc_dv003(high=kl_pd['high'],
                               low=kl_pd['low'],
                               open=kl_pd['open'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv003 = self._format(dv003, name=name)
            impulse_dict[name] = dv003
        return impulse_dict
