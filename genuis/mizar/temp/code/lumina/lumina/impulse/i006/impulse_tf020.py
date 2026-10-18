import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i006.core.tf020 import tf020 as calc_tf020


class ImpulseTf020(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htf020_keys = frozenset(default_keys)  # window, weriod, ewm

    @property
    def name(self):
        return "tf020"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htf020_keys:
            htf020 = calc_tf020(close=kl_pd['close'],
                                open=kl_pd['open'],
                                low=kl_pd['low'],
                                vwap=kl_pd['vwap'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htf020 = self._format(htf020, name=name)
            impulse_dict[name] = htf020
        return impulse_dict
