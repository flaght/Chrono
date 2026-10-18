import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv004 import tv004 as calc_tv004


class ImpulseTv004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv004_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv004_keys:
            htv004 = calc_tv004(close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv004 = self._format(htv004, name=name)
            impulse_dict[name] = htv004
        return impulse_dict
