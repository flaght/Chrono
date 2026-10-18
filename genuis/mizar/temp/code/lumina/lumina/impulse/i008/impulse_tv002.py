import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from lumina.impulse.i008.core.tv002 import tv002 as calc_tv002

class ImpulseTv002(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.htv002_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv002_keys:
            htv002 = calc_tv002(high=kl_pd['high'],
                                low=kl_pd['low'],
                                close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                fast=dk[1],
                                slow=dk[2],
                                ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2],
                                            dk[3])
            htv002 = self._format(htv002, name=name)
            impulse_dict[name] = htv002
        return impulse_dict