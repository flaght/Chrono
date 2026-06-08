import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv020 import tv020 as calc_tv020


class ImpulseTv020(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv020_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv020"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv020_keys:
            htv020 = calc_tv020(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv020 = self._format(htv020, name=name)
            impulse_dict[name] = htv020
        return impulse_dict
