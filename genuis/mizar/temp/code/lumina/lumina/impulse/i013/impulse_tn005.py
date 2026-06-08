from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i013.core.tn005 import tn005 as calc_tn005

class ImpulseTn005(ImpulseBase):
    
        def __init__(self, **kwargs):
            default_keys = default_keys1 if not kwargs else kwargs.get('keys')
            self.tn005_keys = default_keys
    
        @property
        def name(self):
            return "tn005"
    
        def calc_impulse(self, kl_pd):
            impulse_dict = {}
            for dk in self.tn005_keys:
                tn005 = calc_tn005(close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
                name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
                tn005 = self._format(tn005, name=name)
                impulse_dict[name] = tn005
            return impulse_dict