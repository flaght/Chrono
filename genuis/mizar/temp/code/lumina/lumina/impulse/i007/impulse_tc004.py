import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from lumina.impulse.i007.core.tc004 import tc004 as calc_tc004

class ImpulseTc004(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.htc004_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc004"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc004_keys:
            htc004 = calc_tc004(high=kl_pd['high'],
                                low=kl_pd['low'],
                                window=dk[0],
                                fast=dk[1],
                                slow=dk[2],
                                ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3])
            htc004 = self._format(htc004, name=name)
            impulse_dict[name] = htc004
        return impulse_dict