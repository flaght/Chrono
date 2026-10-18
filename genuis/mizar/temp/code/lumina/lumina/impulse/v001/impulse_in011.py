from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.in011 import in011 as calc_in011
import pdb


class ImpulseIn011(ImpulseBase):
    '''
    obv: 过去{}期， 确认价格动能的资金流向累积效应
    '''

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.in011_keys = default_keys

    @property
    def name(self):
        return "in011"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in011_keys:
            in011 = calc_in011(close=kl_pd['close'],
                               volume=kl_pd['volume'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            in011 = self._format(in011,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = in011
        return impulse_dict
