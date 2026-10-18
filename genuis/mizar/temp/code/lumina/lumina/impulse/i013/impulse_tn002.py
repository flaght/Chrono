from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys0
from lumina.impulse.i013.core.tn002 import tn002 as calc_tn002


class ImpulseTn002(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys0 if not kwargs else kwargs.get('keys')
        self.tn002_keys = default_keys

    @property
    def name(self):
        return "tn002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.tn002_keys:
            tn002 = calc_tn002(close=kl_pd['close'],
                               window=dk[0],
                               ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            tn002 = self._format(tn002, name=name)
            impulse_dict[name] = tn002
        return impulse_dict
