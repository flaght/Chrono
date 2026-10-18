import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv004 import dv004 as calc_dv004


class ImpulseDv004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv004_keys = default_keys

    @property
    def name(self):
        return "dv004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv004_keys:
            dv004 = calc_dv004(high=kl_pd['high'],
                               low=kl_pd['low'],
                               close=kl_pd['close'],
                               open=kl_pd['open'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv004 = self._format(dv004, name=name)
            impulse_dict[name] = dv004
        return impulse_dict
