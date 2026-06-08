from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.ht006 import ht006 as calc_ht006
import pdb


class ImpulseHt006(ImpulseBase):
    """
    KDJ 过去{}期{}数据
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht006_keys = default_keys

    @property
    def name(self):
        return "ht006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht006_keys:
            k, d, j = calc_ht006(close=kl_pd[dk[0]],
                               window=dk[1],
                               weriod=dk[2],
                               ewm=True if dk[3] == 1 else False)
            
            name_k = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3],
                                              'k')
            name_d = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3],
                                              'd')
            name_j = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1], dk[2], dk[3],
                                              'j')
            k = self._format(k,
                             name=name_k,
                             desc=self.__class__.__doc__.format(dk[1],dk[0]))
            d = self._format(d,
                             name=name_d,
                             desc=self.__class__.__doc__.format(dk[1],dk[0]))
            j = self._format(j,
                             name=name_j,
                             desc=self.__class__.__doc__.format(dk[1],dk[0]))

            impulse_dict[name_k] = k
            impulse_dict[name_d] = d
            impulse_dict[name_j] = j
        return impulse_dict

