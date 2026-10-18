import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv008 import iv008 as calc_tv008

class ImpulseIv008(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv008_keys = default_keys

    @property
    def name(self):
        return "iv008"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv008_keys:
            iv008 = calc_tv008(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv008 = self._format(iv008, name=name)
            impulse_dict[name] = iv008
        return impulse_dict