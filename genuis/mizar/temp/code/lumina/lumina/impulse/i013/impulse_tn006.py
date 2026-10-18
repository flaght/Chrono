from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i013.core.tn006 import tn006 as calc_tn006

class ImpulseTn006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.tn006_keys = default_keys

    @property
    def name(self):
        return "tn006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.tn006_keys:
            tn006 = calc_tn006(close=kl_pd['close'],
                               volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            tn006 = self._format(tn006, name=name)
            impulse_dict[name] = tn006
        return impulse_dict