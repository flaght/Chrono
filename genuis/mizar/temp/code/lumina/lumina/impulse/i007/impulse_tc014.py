import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from lumina.impulse.i007.core.tc014 import tc014 as calc_tc014

class ImpulseTc014(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.htc014_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc014"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc014_keys:
            htc014 = calc_tc014(close=kl_pd['close'],
                                window=dk[0],
                                fast=dk[1],
                                slow=dk[2],
                                ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3])
            htc014 = self._format(htc014, name=name)
            impulse_dict[name] = htc014
        return impulse_dict