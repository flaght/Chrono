import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv010 import dv010 as calc_dv010


class ImpulseDv010(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv010_keys = default_keys

    @property
    def name(self):
        return "dv010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv010_keys:
            dv010 = calc_dv010(close=kl_pd['close'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv010 = self._format(dv010, name=name)
            impulse_dict[name] = dv010
        return impulse_dict
