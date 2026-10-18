import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i007.core.tc011 import tc011 as calc_tc011

class ImpulseTc011(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htc011_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc011"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc011_keys:
            htc011 = calc_tc011(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htc011 = self._format(htc011, name=name)
            impulse_dict[name] = htc011
        return impulse_dict