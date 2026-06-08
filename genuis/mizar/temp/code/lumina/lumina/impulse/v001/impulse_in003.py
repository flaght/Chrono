from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.in003 import in003 as calc_in003
import pdb


class ImpulseIn003(ImpulseBase):
    '''
    EMA: {} 期指数移动平均线
    '''

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.in003_keys = default_keys

    @property
    def name(self):
        return "in003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in003_keys:
            in003 = calc_in003(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            in003 = self._format(in003,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = in003
        return impulse_dict
