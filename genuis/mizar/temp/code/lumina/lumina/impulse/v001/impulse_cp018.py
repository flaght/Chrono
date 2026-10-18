from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp018 import cp018 as calc_cp018

class ImpulseCp018(ImpulseBase):
    """
    {}期支撑与最近阻力的强度比。大于1说明下方支撑强于上方阻力，价格易涨难跌。
    """
    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp018"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp018(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha, name=name, desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict