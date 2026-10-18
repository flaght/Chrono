import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i009.core.iv012 import iv012 as calc_tv012


class ImpulseIv012(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.iv012_keys = default_keys

    @property
    def name(self):
        return "iv012"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.iv012_keys:
            iv012 = calc_tv012(close=kl_pd['close'],
                               volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            iv012 = self._format(iv012, name=name)
            impulse_dict[name] = iv012
        return impulse_dict
