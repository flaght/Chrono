import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv005 import iv005 as calc_tv005

class ImpulseIv005(ImpulseBase):
    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv005_keys = default_keys
    
    @property
    def name(self):
        return "iv005"
    
    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv005_keys:
            iv005 = calc_tv005(value=kl_pd['value'],
                                volume=kl_pd['volume'],
                                close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv005 = self._format(iv005, name=name)
            impulse_dict[name] = iv005
        return impulse_dict