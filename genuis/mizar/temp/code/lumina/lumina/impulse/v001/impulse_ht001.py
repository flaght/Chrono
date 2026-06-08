from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.ht001 import ht001 as calc_ht001
import pdb


class ImpulseHt001(ImpulseBase):
    """
    SMA  {}期{}数据简单移动平均线
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht001_keys = default_keys
        
    @property
    def name(self):
        return "ht001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht001_keys:
            ht001 = calc_ht001(close=kl_pd[dk[0]],
                               window=dk[1],
                               weriod=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[1], dk[2], dk[3])
            ht001 = self._format(ht001,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1], dk[0]))
            impulse_dict[name] = ht001
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.ht001_keys[0][1], self.ht001_keys[0][0])
