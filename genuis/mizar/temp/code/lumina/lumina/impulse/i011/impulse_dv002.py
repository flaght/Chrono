import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv002 import dv002 as calc_dv002


class ImpulseDv002(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv002_keys = default_keys

    @property
    def name(self):
        return "dv002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv002_keys:
            dv002 = calc_dv002(close=kl_pd['close'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv002 = self._format(dv002, name=name)
            impulse_dict[name] = dv002
        return impulse_dict
