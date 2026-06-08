from lumina.impulse.base import ImpulseBase
from lumina.impulse.v001.core.cp005 import cp005 as calc_cp005


class ImpulseCp005(ImpulseBase):
    """
    筹码分布中显著峰值的数量。单峰密集为佳；多峰形态意味着存在多个套牢盘，上行阻力重重
    """

    def __init__(self, **kwargs):
        self.cp_keys = kwargs.get('keys', [(0.8, 5)])

    @property
    def name(self):
        return "cp005"

    def calc_impulse(self, kl_pd):
        impulse_dict = {}
        for dk in self.cp_keys:
            alpha = calc_cp005(chip_data=kl_pd['chip_data'],
                               close=kl_pd['close'],
                               chip_peak_strength=dk[0],
                               distance=dk[1])
            name = f"{self.name}_{dk[0]}_{dk[1]}"
            alpha = self._format(alpha,
                                 name=name,
                                 desc=self.__class__.__doc__.format(dk[0]))
            impulse_dict[name] = alpha
        return impulse_dict
