from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.ht002 import ht002 as calc_ht002
import pdb


class ImpulseHt002(ImpulseBase):
    """
    EMA  {}期{}数据指数移动平均线
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.ht002_keys = default_keys

    @property
    def name(self):
        return "ht002"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.ht002_keys:
            ht002 = calc_ht002(close=kl_pd[dk[0]],
                               window=dk[1],
                               weriod=dk[2],
                               ewm=True if dk[3] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[1], dk[2], dk[3])
            ht002 = self._format(ht002,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1], dk[0]))
            impulse_dict[name] = ht002
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.ht002_keys[0][1], self.ht002_keys[0][0])
