from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from lumina.impulse.i013.core.tn003 import tn003 as calc_tn003


class ImpulseTn003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.tn003_keys = default_keys

    @property
    def name(self):
        return "tn003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.tn003_keys:
            tn003 = calc_tn003(close=kl_pd['close'],
                               long=dk[1],
                               medium=dk[2],
                               short=dk[3],
                               window=dk[0],
                               ewm=True if dk[4] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1],
                                                    dk[2], dk[3], dk[4])
            tn003 = self._format(tn003, name=name)
            impulse_dict[name] = tn003
        return impulse_dict
