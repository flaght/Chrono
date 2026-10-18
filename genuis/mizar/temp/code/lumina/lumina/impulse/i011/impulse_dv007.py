import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv007 import dv007 as calc_dv007


class ImpulseDv007(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv007_keys = default_keys

    @property
    def name(self):
        return "dv007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv007_keys:
            dv007 = calc_dv007(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv007 = self._format(dv007, name=name)
            impulse_dict[name] = dv007
        return impulse_dict
