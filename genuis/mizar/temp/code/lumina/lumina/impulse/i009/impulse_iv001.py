import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv001 import iv001 as calc_tv001

class ImpulseIv001(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv001_keys = default_keys

    @property
    def name(self):
        return "iv001"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv001_keys:
            iv001 = calc_tv001(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv001 = self._format(iv001, name=name)
            impulse_dict[name] = iv001
        return impulse_dict