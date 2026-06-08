import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv010 import iv010 as calc_tv010

class ImpulseIv010(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv010_keys = default_keys

    @property
    def name(self):
        return "iv010"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv010_keys:
            iv010 = calc_tv010(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv010 = self._format(iv010, name=name)
            impulse_dict[name] = iv010
        return impulse_dict