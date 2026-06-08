import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv014 import tv014 as calc_tv014


class ImpulseTv014(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv014_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv014"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv014_keys:
            htv014 = calc_tv014(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv014 = self._format(htv014, name=name)
            impulse_dict[name] = htv014
        return impulse_dict
