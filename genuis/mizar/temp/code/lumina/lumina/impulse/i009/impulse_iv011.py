import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv011 import iv011 as calc_tv011

class ImpulseIv011(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv011_keys = default_keys

    @property
    def name(self):
        return "iv011"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv011_keys:
            iv011 = calc_tv011(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv011 = self._format(iv011, name=name)
            impulse_dict[name] = iv011
        return impulse_dict