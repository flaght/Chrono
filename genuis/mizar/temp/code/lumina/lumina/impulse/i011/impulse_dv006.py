import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv006 import dv006 as calc_dv006


class ImpulseDv006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv006_keys = default_keys

    @property
    def name(self):
        return "dv006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv006_keys:
            dv006 = calc_dv006(high=kl_pd['high'],
                               low=kl_pd['low'],
                               close=kl_pd['close'],
                               open=kl_pd['open'],
                               volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv006 = self._format(dv006, name=name)
            impulse_dict[name] = dv006
        return impulse_dict
