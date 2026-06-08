from lumina.impulse.base import ImpulseBase
from lumina.impulse.base import default_key4
from lumina.impulse.v001.core.mo003 import mo003 as calc_mo003


class ImpulseMo003(ImpulseBase):
    """期货会员持仓多头仓和空头仓差值的变化率，即当期(多头仓空头仓差值)除以{0}期 前(多头仓空头仓差值)"""

    def __init__(self, **kwargs):
        default_keys = default_key4 if not kwargs else kwargs.get('keys')
        self.mo003_keys = default_keys

    @property
    def name(self):
        return "mo003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.mo003_keys:
            mo003 = calc_mo003(long=kl_pd['long'],
                               short=kl_pd['short'],
                               window=dk[0],
                               weriod=dk[1],
                               ewm=True if dk[2] == 1 else False)
            name = "{0}_{1}_{2}_{3}".format(self.name, dk[0], dk[1], dk[2])
            mo003 = self._format(mo003,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = mo003
        return impulse_dict

    def description(self, base_str=None):
        base_str = self.__class__.__doc__ if base_str is None else base_str
        return base_str.format(self.mo003_keys[0][1])
