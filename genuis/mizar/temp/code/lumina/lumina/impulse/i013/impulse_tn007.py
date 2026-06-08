import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i013.core.tn007 import tn007 as calc_tn007


class ImpulseTn007(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.tn007_keys = default_keys

    @property
    def name(self):
        return "tn007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.tn007_keys:
            tn007 = calc_tn007(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            tn007 = self._format(tn007, name=name)
            impulse_dict[name] = tn007
        return impulse_dict
