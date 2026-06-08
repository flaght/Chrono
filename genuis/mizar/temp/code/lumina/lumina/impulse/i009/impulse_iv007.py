import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv007 import iv007 as calc_tv007


class ImpulseIv007(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv007_keys = default_keys

    @property
    def name(self):
        return "iv007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv007_keys:
            iv007 = calc_tv007(close=kl_pd['value'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv007 = self._format(iv007, name=name)
            impulse_dict[name] = iv007
        return impulse_dict
