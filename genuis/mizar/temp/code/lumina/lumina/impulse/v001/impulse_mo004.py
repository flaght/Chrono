from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mo004 import mo004 as calc_mo004


class ImpulseMo004(ImpulseBase):
    """期货会员持多头仓{}期的变化率"""

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mo004_keys = default_keys

    @property
    def name(self):
        return "mo004"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mo004_keys:
            mo004 = calc_mo004(long=kl_pd['long'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            mo004 = self._format(mo004,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mo004
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mo004_keys[0][1])
