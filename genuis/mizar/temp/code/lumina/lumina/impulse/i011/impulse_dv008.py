import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv008 import dv008 as calc_dv008


class ImpulseDv008(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv008_keys = default_keys

    @property
    def name(self):
        return "dv008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv008_keys:
            dv008 = calc_dv008(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv008 = self._format(dv008, name=name)
            impulse_dict[name] = dv008
        return impulse_dict
