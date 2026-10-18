import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from lumina.impulse.i007.core.tc009 import tc009 as calc_tc009


class ImpulseTc009(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.htc009_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc009_keys:
            htc009 = calc_tc009(close=kl_pd['close'],
                                window=dk[0],
                                fast=dk[1],
                                slow=dk[2],
                                weriod=dk[3],
                                ewm=True if dk[4] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1],
                                                    dk[2], dk[3], dk[4])
            htc009 = self._format(htc009, name=name)
            impulse_dict[name] = htc009
        return impulse_dict
