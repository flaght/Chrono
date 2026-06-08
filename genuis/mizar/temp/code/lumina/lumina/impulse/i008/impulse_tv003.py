import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv003 import tv003 as calc_tv003


class ImpulseTv003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv003_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv003_keys:
            htv003 = calc_tv003(high=kl_pd['high'],
                                low=kl_pd['low'],
                                close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htv003 = self._format(htv003, name=name)
            impulse_dict[name] = htv003
        return impulse_dict
