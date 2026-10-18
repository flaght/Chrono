from lumina.impulse.base import ImpulseBase
from lumina.impulse.v001.core.cp003 import cp003 as calc_cp003


class ImpulseCp003(ImpulseBase):
    """
    筹码最密集的{}价格区间的总筹码占比。值越高，说明筹码锁定性越好，主力控盘度可能越高
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', [(0.1)])

    @property
    def name(self):
        return "cp003"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp003(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               percent=dk[0])
            name = f"{self.name}_{dk[0] * 100}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
