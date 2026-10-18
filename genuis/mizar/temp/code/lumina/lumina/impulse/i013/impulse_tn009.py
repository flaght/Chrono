import pdb
from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_keys1
from lumina.impulse.i013.core.tn009 import tn009 as calc_tn009


class ImpulseTn009(ImpulseBase):

    def __init__(self, **kwargs):
        default_keys = default_keys1 if not kwargs else kwargs.get('keys')
        self.tn009_keys = default_keys

    @property
    def name(self):
        return "tn009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.tn009_keys:
            tn009_1, tn009_2, tn009_3, tn009_4 = calc_tn009(
                low=kl_pd['low'],
                high=kl_pd['high'],
                volume=kl_pd['volume'],
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
            tn009_1 = self._format(tn009_1, name=name1)
            tn009_2 = self._format(tn009_2, name=name2)
            tn009_3 = self._format(tn009_3, name=name3)
            tn009_4 = self._format(tn009_4, name=name4)
            impulse_dict[name1] = tn009_1
            impulse_dict[name2] = tn009_2
            impulse_dict[name3] = tn009_3
            impulse_dict[name4] = tn009_4

        return impulse_dict
