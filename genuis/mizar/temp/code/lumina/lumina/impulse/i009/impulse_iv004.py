import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys3
from lumina.impulse.i009.core.iv004 import iv004 as calc_tv004


class ImpulseIv004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys3 if not kwargs else kwargs.get('keys')
        self.iv004_keys = default_keys

    @property
    def name(self):
        return "iv004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv004_keys:
            iv004 = calc_tv004(close=kl_pd['close'],
                               window=dk[0],
                               fast=dk[1],
                               slow=dk[2],
                               weriod=dk[3],
                               ewm=True if dk[4] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1],
                                                    dk[2], dk[3], dk[4])
            iv004 = self._format(iv004, name=name)
            impulse_dict[name] = iv004
        return impulse_dict
