import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i013.core.tn008 import tn008 as calc_tn008


class ImpulseTn008(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.tn008_keys = default_keys

    @property
    def name(self):
        return "tn008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.tn008_keys:
            tn008_1, tn008_2, tn008_3, tn008_4 = calc_tn008(
                low=kl_pd['low'],
                high=kl_pd['high'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False)
            name1 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 1)
            name2 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 2)
            name3 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 3)
            name4 = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                 dk[2], 4)
            tn008_1 = self._format(tn008_1, name=name1)
            tn008_2 = self._format(tn008_2, name=name2)
            tn008_3 = self._format(tn008_3, name=name3)
            tn008_4 = self._format(tn008_4, name=name4)
            impulse_dict[name1] = tn008_1
            impulse_dict[name2] = tn008_2
            impulse_dict[name3] = tn008_3
            impulse_dict[name4] = tn008_4

        return impulse_dict
