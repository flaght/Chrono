from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.in008 import in008 as calc_in008
import pdb


class ImpulseIn008(ImpulseBase):
    '''
    ATR: 过去{}期
    '''

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.in008_keys = default_keys

    @property
    def name(self):
        return "in008"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in008_keys:
            in008 = calc_in008(high=kl_pd['high'],
                               low=kl_pd['low'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            in008 = self._format(in008,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = in008
        return impulse_dict
