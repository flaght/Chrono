import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.a001.core.al003 import al003 as calc_al003


class ImpulseAl003(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.rv001_keys = default_keys

    @property
    def name(self):
        return "al003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.rv001_keys:
            al003 = calc_al003(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2],
                                                1)
            al003 = self._format(al003, name=name)
            impulse_dict[name] = al003
        return impulse_dict
