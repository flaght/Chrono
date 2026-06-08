import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i006.core.tf006 import tf006 as calc_tf006

class ImpulseTf006(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htf006_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "tf006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htf006_keys:
            htf006 = calc_tf006(high=kl_pd['high'],
                                value=kl_pd['value'] / 1e6,
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htf006 = self._format(htf006, name=name)
            impulse_dict[name] = htf006
        return impulse_dict