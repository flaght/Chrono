from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.in004 import in004 as calc_in004
import pdb


class ImpulseIn004(ImpulseBase):
    '''
    RSI:{} 期涨跌幅度的比率评估市场多空力量对比
    '''

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.in004_keys = default_keys

    @property
    def name(self):
        return "in004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.in004_keys:
            in004 = calc_in004(close=kl_pd['close'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            in004 = self._format(in004,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = in004
        return impulse_dict
