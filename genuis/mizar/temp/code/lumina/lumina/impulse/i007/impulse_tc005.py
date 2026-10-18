import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from lumina.impulse.i007.core.tc005 import tc005 as calc_tc005


class ImpulseTc005(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.htc005_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc005_keys:
            htc005 = calc_tc005(close=kl_pd['close'],
                                window=dk[0],
                                fast=dk[1],
                                slow=dk[2],
                                ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2],
                                                dk[3])
            htc005 = self._format(htc005, name=name)
            impulse_dict[name] = htc005
        return impulse_dict
