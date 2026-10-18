import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i011.core.dv012 import dv012 as calc_dv012


class ImpulseDv012(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.dv012_keys = default_keys

    @property
    def name(self):
        return "dv012"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.dv012_keys:
            dv012 = calc_dv012(close=kl_pd['close'],
                               high=kl_pd['high'],
                               low=kl_pd['low'],
                               open=kl_pd['open'],
                               value=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            dv012 = self._format(dv012, name=name)
            impulse_dict[name] = dv012
        return impulse_dict
