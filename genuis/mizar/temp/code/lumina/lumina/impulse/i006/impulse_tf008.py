import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i006.core.tf008 import tf008 as calc_tf008


class ImpulseTf008(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htf008_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "tf008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htf008_keys:
            htf008 = calc_tf008(high=kl_pd['high'],
                                open=kl_pd['open'],
                                volume=kl_pd['volume'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htf008 = self._format(htf008, name=name)
            impulse_dict[name] = htf008
        return impulse_dict
