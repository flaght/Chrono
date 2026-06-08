from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp011 import cp011 as calc_cp011

class ImpulseCp011(ImpulseBase):
    """
    {}期均价与最强筹码峰的偏离程度。正值越大，代表股价已有效脱离主要成本区。
    """
    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp011"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp011(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha, name=name, desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict