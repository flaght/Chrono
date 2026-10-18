import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i006.core.tf002 import tf002 as calc_tf002


class ImpulseTf002(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htf002_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "tf002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htf002_keys:
            htf002 = calc_tf002(open=kl_pd['open'],
                                high=kl_pd['high'],
                                close=kl_pd['close'],
                                volume=kl_pd['volume'],
                                value=kl_pd['value'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htf002 = self._format(htf002, name=name)
            impulse_dict[name] = htf002
        return impulse_dict
