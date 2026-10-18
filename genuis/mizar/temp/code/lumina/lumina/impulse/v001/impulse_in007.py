from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.in007 import in007 as calc_in007
import pdb


class ImpulseIn007(ImpulseBase):
    '''
    KDJ 过去{}期
    '''

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.in007_keys = default_keys

    @property
    def name(self):
        return "in007"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in007_keys:
            k, d, j = calc_in007(close=kl_pd['close'],
                                 window=dk[0],
                                 weriod=dk[1],
                                 ewm=True if dk[2] == 1 else False)
            name_k = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2],
                                              'k')
            name_d = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2],
                                              'd')
            name_j = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2],
                                              'j')
            k = self._format(k,
                             name=name_k,
                             desc=self.__class__.__doc__.format(dk[1]))
            d = self._format(d,
                             name=name_d,
                             desc=self.__class__.__doc__.format(dk[1]))
            j = self._format(j,
                             name=name_j,
                             desc=self.__class__.__doc__.format(dk[1]))

            impulse_dict[name_k] = k
            impulse_dict[name_d] = d
            impulse_dict[name_j] = j
        return impulse_dict
