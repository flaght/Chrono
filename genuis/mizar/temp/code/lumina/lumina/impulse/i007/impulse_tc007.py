import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i007.core.tc007 import tc007 as calc_tc007

class ImpulseTc007(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htc007_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc007"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc007_keys:
            htc007 = calc_tc007(high=kl_pd['high'],
                                low=kl_pd['low'],
                                close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htc007 = self._format(htc007, name=name)
            impulse_dict[name] = htc007
        return impulse_dict