from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.in009 import in009 as calc_in009
import pdb


class ImpulseIn009(ImpulseBase):
    '''
     VWAP: {0}期 (成交量 * 收盘价)之和 / {0}期成交量之和 判断价格相对当日成交重心的强弱
    '''

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.in009_keys = default_keys

    @property
    def name(self):
        return "in009"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in009_keys:
            in009 = calc_in009(volume=kl_pd['volume'],
                               close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            in009 = self._format(in009,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = in009
        return impulse_dict
