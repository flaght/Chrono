from lumina.impulse.base import ImpulseBase, default_key7
from lumina.impulse.v001.core.cp015 import cp015 as calc_cp015


class ImpulseCp015(ImpulseBase):
    """
    {}期收盘价所在区间的筹码密度。值越高，说明当前价位是多空争夺焦点，正在激烈换手
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', default_key7)

    @property
    def name(self):
        return "cp015"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp015(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               weriod=dk[0],
                               ewm=(dk[1] == 1))
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
