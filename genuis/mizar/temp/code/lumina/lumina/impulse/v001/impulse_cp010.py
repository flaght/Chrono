from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp010 import cp010 as calc_cp010


class ImpulseCp010(ImpulseBase):
    """
    {}期均获利盘总量与套牢盘总量的比值。远大于1，多头主导；远小于1，空头（套牢盘）主导。
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp010"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp010(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[1]))
            impulse_dict[name] = alpha
        return impulse_dict
