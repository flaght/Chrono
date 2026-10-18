from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp016 import cp016 as calc_cp016


class ImpulseCp016(ImpulseBase):
    """
    {}期N期平均价格下方第一个显著筹码峰的强度。量化了股价回调时，最近的“护城河”有多深。
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp016"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp016(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
