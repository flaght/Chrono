import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys2
from lumina.impulse.i008.core.tv022 import tv022 as calc_tv022


class ImpulseTv022(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys2 if not kwargs else kwargs.get('keys')
        self.htv022_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv022"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv022_keys:
            htv022 = calc_tv022(high=kl_pd['high'],
                                low=kl_pd['low'],
                                window=dk[0],
                                tenkan=dk[1],
                                kijun=dk[2],
                                ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2],
                                                dk[3])
            htv022 = self._format(htv022, name=name)
            impulse_dict[name] = htv022
        return impulse_dict
