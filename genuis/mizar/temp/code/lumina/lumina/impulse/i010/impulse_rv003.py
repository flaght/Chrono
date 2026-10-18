import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i010.core.rv003 import rv003 as calc_rv003

class ImpulseRv003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv003_keys = default_keys

    @property
    def name(self):
        return "rv003"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv003_keys:
            rv003_1, rv003_2 = calc_rv003(value=kl_pd['value'],
                                          volume=kl_pd['volume'],
                                          window=dk[0],
                                          weriod=dk[1],
                                          ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 2)
            rv003_1 = self._format(rv003_1, name=name1)
            rv003_2 = self._format(rv003_2, name=name2)
            impulse_dict[name1] = rv003_1
            impulse_dict[name2] = rv003_2
        return impulse_dict
