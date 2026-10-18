import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i006.core.tf004 import tf004 as calc_tf004


class ImpulseTf004(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htf004_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "tf004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htf004_keys:
            htf004 = calc_tf004(low=kl_pd['low'],
                                value=kl_pd['value'] / 1e6,
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htf004 = self._format(htf004, name=name)
            impulse_dict[name] = htf004
        return impulse_dict
