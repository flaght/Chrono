import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i007.core.tc020 import tc020 as calc_tc020


class ImpulseTc020(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.htc020_keys = frozenset(default_keys)

    @property
    def name(self):
        return "tc020"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.htc020_keys:
            htc020 = calc_tc020(close=kl_pd['close'],
                                window=dk[0],
                                weriod=dk[1],
                                ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            htc020 = self._format(htc020, name=name)
            impulse_dict[name] = htc020
        return impulse_dict
