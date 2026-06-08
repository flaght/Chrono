import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.a001.core.al001 import al001 as calc_al001


class ImpulseAl001(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv001_keys = default_keys

    @property
    def name(self):
        return "al001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv001_keys:
            al001 = calc_al001(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2],
                                                1)
            al001 = self._format(al001, name=name)
            impulse_dict[name] = al001
        return impulse_dict
