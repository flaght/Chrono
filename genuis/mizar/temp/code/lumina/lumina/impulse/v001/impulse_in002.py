from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.in002 import in002 as calc_in002
import pdb


class ImpulseIn002(ImpulseBase):
    '''
    SMA  {}期简单移动平均线
    '''

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.in002_keys = default_keys

    @property
    def name(self):
        return "in002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in002_keys:
            in002 = calc_in002(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            in002 = self._format(in002,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = in002
        return impulse_dict
