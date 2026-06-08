import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i007.core.tc022 import tc022 as calc_tc022


class ImpulseTc022(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htc022_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc022"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc022_keys:
            htc022 = calc_tc022(close=kl_pd['close'],
                                high=kl_pd['high'],
                                low=kl_pd['low'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htc022 = self._format(htc022, name=name)
            impulse_dict[name] = htc022
        return impulse_dict
