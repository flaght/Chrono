import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i012.core.oi009 import oi009 as calc_oi009


class ImpulseOi009(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.oi009_keys = default_keys

    @property
    def name(self):
        return "oi009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.oi009_keys:
            oi009 = calc_oi009(close=kl_pd['close'],
                               openint=kl_pd['openint'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            oi009 = self._format(oi009, name=name)
            impulse_dict[name] = oi009
        return impulse_dict
