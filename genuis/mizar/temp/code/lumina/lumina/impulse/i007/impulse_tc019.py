import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i007.core.tc019 import tc019 as calc_tc019

class ImpulseTc019(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htc019_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc019"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc019_keys:
            htc019 = calc_tc019(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htc019 = self._format(htc019, name=name)
            impulse_dict[name] = htc019
        return impulse_dict