import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i006.core.tf003 import tf003 as calc_tf003

class ImpulseTf003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htf003_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "tf003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htf003_keys:
            htf003 = calc_tf003(close=kl_pd['close'],
                                vwap=kl_pd['vwap'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htf003 = self._format(htf003, name=name)
            impulse_dict[name] = htf003
        return impulse_dict