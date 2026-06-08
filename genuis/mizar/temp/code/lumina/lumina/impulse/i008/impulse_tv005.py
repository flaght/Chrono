import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv005 import tv005 as calc_tv005


class ImpulseTv005(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv005_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv005_keys:
            htv005 = calc_tv005(high=kl_pd['high'],
                                low=kl_pd['low'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv005 = self._format(htv005, name=name)
            impulse_dict[name] = htv005
        return impulse_dict
