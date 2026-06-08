from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mo001 import mo001 as calc_mo001
import pdb


class ImpulseMo001(ImpulseBase):
    """
    期货会员持多头仓 {} 期的变化
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mo001_keys = default_keys

    @property
    def name(self):
        return "mo001"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mo001_keys:
            mo001 = calc_mo001(long=kl_pd['long'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            mo001 = self._format(mo001,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mo001
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mo001_keys[0][1])
