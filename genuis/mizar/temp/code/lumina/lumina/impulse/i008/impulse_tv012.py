import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys0
from lumina.impulse.i008.core.tv012 import tv012 as calc_tv012


class ImpulseTv012(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys0 if not kwargs else kwargs.get('keys')
        self.htv012_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tv012"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htv012_keys:
            htv012 = calc_tv012(open=kl_pd['open'],
                                high=kl_pd['high'],
                                low=kl_pd['low'],
                                close=kl_pd['close'],
                                window=dk[0],
                                ewm=True if dk[1] == 1 else False)
            name = "{0}_{1}_{2}".format(self.name, dk[0], dk[1])
            htv012 = self._format(htv012, name=name)
            impulse_dict[name] = htv012
        return impulse_dict
