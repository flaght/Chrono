import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv011 import dv011 as calc_dv011


class ImpulseDv011(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv011_keys = default_keys

    @property
    def name(self):
        return "dv011"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv011_keys:
            dv011 = calc_dv011(close=kl_pd['close'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv011 = self._format(dv011, name=name)
            impulse_dict[name] = dv011
        return impulse_dict
