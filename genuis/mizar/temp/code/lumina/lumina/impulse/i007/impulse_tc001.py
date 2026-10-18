import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i007.core.tc001 import tc001 as calc_tc001

class ImpulseTc001(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htc001_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc001"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc001_keys:
            htc001 = calc_tc001(volume=kl_pd['volume'] / 1e6,
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htc001 = self._format(htc001, name=name)
            impulse_dict[name] = htc001
        return impulse_dict