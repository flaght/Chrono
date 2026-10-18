import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv019 import tv019 as calc_tv019


class ImpulseTv019(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv019_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv019"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv019_keys:
            htv019 = calc_tv019(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv019 = self._format(htv019, name=name)
            impulse_dict[name] = htv019
        return impulse_dict
