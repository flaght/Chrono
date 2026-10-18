import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i007.core.tc002 import tc002 as calc_tc002

class ImpulseTc002(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htc002_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc002"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc002_keys:
            htc002 = calc_tc002(volume=kl_pd['volume'] / 1e6,
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htc002 = self._format(htc002, name=name)
            impulse_dict[name] = htc002
        return impulse_dict