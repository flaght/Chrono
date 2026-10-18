from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.ht005 import ht005 as calc_ht005
import pdb


class ImpulseHt005(ImpulseBase):
    """
    Bollinger Bands {}期{}数据布林带
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht005_keys = default_keys

    @property
    def name(self):
        return "ht005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht005_keys:
            middle, upper, lower = calc_ht005(close=kl_pd[dk[0]],
                               window=dk[1],
                               weriod=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name_middle = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1],
                                                       dk[2], dk[3], "middle")
            name_upper = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1],
                                                      dk[2], dk[3], "upper")
            name_lower = "{0}_{1}_{2}_{3}_{4}_{5}".format(self.name, dk[0], dk[1],
                                                      dk[2], dk[3], "lower")
            middle = self._format(middle,
                                name=name_middle,
                                desc=self.__class__.__doc__.format(
                                    dk[1],dk[0]))
            upper = self._format(upper,
                                  name=name_upper,
                                  desc=self.__class__.__doc__.format(
                                      dk[1], dk[0]))
            lower = self._format(lower,
                                 name=name_lower,
                                 desc=self.__class__.__doc__.format(
                                     dk[1], dk[0]))
            impulse_dict[name_middle] = middle
            impulse_dict[name_upper] = upper
            impulse_dict[name_lower] = lower
        return impulse_dict
