import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i007.core.tc021 import tc021 as calc_tc021


class ImpulseTc021(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htc021_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc021"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc021_keys:
            htc021 = calc_tc021(close=kl_pd['close'],
                                volume=kl_pd['volume'] / 1e6,
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htc021 = self._format(htc021, name=name)
            impulse_dict[name] = htc021
        return impulse_dict
