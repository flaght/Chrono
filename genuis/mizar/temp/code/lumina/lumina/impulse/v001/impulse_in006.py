from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.in006 import in006 as calc_in006
import pdb


class ImpulseIn006(ImpulseBase):
    '''
    Bollinger Bands {}期布林带
    '''

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.in006_keys = default_keys

    @property
    def name(self):
        return "in006"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in006_keys:
            middle, upper, lower = calc_in006(
                close=kl_pd['close'],
                window=dk[0],
                weriod=dk[1],
                ewm=True if dk[2] == 1 else False)
            name_middle = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                       dk[2], "middle")
            name_upper = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                      dk[2], "upper")
            name_lower = "{0}_{1}_{2}_{3}_{4}".format(self.name, dk[0], dk[1],
                                                      dk[2], "lower")
            middle = self._format(middle,
                                  name=name_middle,
                                  desc=self.__class__.__doc__.format(dk[1]))
            upper = self._format(upper,
                                 name=name_upper,
                                 desc=self.__class__.__doc__.format(dk[1]))
            lower = self._format(lower,
                                 name=name_lower,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name_middle] = middle
            impulse_dict[name_upper] = upper
            impulse_dict[name_lower] = lower
        return impulse_dict
