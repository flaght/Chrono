import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv006 import tv006 as calc_tv006


class ImpulseTv006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv006_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv006_keys:
            htv006 = calc_tv006(close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv006 = self._format(htv006, name=name)
            impulse_dict[name] = htv006
        return impulse_dict
