from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i013.core.tn004 import tn004 as calc_tn004


class ImpulseTn004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.tn004_keys = default_keys

    @property
    def name(self):
        return "tn004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.tn004_keys:
            tn004 = calc_tn004(close=kl_pd['close'],
                               open=kl_pd['open'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            tn004 = self._format(tn004, name=name)
            impulse_dict[name] = tn004
        return impulse_dict
