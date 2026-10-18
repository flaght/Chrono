from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mo005 import mo005 as calc_mo005


class ImpulseMo005(ImpulseBase):
    """
    期货会员持空头仓{}的变化率
    """

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mo005_keys = default_keys

    @property
    def name(self):
        return "mo005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mo005_keys:
            mo005 = calc_mo005(short=kl_pd['short'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            mo005 = self._format(mo005,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mo005
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mo005_keys[0][1])
