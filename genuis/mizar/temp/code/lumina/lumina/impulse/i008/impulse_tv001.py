import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i008.core.tv001 import tv001 as calc_tv001

class ImpulseTv001(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htv001_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv001_keys:
            htv001 = calc_tv001(high=kl_pd['high'],
                                low=kl_pd['low'],
                                close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            htv001 = self._format(htv001, name=name)
            impulse_dict[name] = htv001
        return impulse_dict