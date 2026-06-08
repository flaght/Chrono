from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp017 import cp017 as calc_cp017


class ImpulseCp017(ImpulseBase):
    """
    {}期价格上方第一个显著筹码峰的强度。量化了股价上涨时，即将遇到的第一个“拦路虎”有多强。
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp017"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp017(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
