import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv021 import tv021 as calc_tv021


class ImpulseTv021(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv021_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv021"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv021_keys:
            htv021 = calc_tv021(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv021 = self._format(htv021, name=name)
            impulse_dict[name] = htv021
        return impulse_dict
