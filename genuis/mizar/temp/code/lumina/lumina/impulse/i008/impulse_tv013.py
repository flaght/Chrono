import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv013 import tv013 as calc_tv013


class ImpulseTv013(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv013_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv013"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv013_keys:
            htv013 = calc_tv013(high=kl_pd['high'],
                                low=kl_pd['low'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv013 = self._format(htv013, name=name)
            impulse_dict[name] = htv013
        return impulse_dict
